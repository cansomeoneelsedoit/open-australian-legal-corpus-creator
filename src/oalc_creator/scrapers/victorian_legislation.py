import re
import json
import asyncio

from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import lxml.html

from inscriptis.css_profiles import CSS_PROFILES
from inscriptis.model.html_element import HtmlElement
from inscriptis.html_properties import Display
from zipfile import BadZipFile

from ..data import Entry, Request, Document, make_doc
from ..helpers import log, warning
from ..scraper import Scraper
from ..custom_mammoth import docx2html
from ..custom_inscriptis import CustomInscriptis, CustomParserConfig


class VictorianLegislation(Scraper):
    """A scraper for the Victorian Legislation database.

    The database is a Nuxt application backed by the Tide (Drupal) content API.
    In-force documents are enumerated from the site's sitemap, each document's
    landing page embeds a `__NUXT_DATA__` payload that identifies the current
    consolidated version, and the `/api/tide/in-force` endpoint resolves that
    version to a DOCX download."""

    def __init__(self,
                 indices_refresh_interval: bool | timedelta = None,
                 index_refresh_interval: bool | timedelta = None,
                 semaphore: asyncio.Semaphore = None,
                 session: aiohttp.ClientSession = None,
                 thread_pool_executor: ThreadPoolExecutor = None,
                 ocr_semaphore: asyncio.Semaphore = None,
                 ) -> None:
        super().__init__(
            source='victorian_legislation',
            indices_refresh_interval=indices_refresh_interval,
            index_refresh_interval=index_refresh_interval,
            semaphore=semaphore or asyncio.Semaphore(10), # Employ a lower semaphore limit to avoid overloading the database.
            session=session,
            thread_pool_executor=thread_pool_executor,
            ocr_semaphore=ocr_semaphore
        )

        self._jurisdiction = 'victoria'
        self._base_url = 'https://www.legislation.vic.gov.au'

        self._uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

        # Create a custom Inscriptis CSS profile.
        inscriptis_profile = CSS_PROFILES['strict'].copy()
        inscriptis_profile['p'] = HtmlElement(display=Display.block)
        inscriptis_profile |= dict.fromkeys(('h1', 'h2', 'h3', 'h4', 'h5'), HtmlElement(display=Display.block, margin_before=1))

        self._inscriptis_config = CustomParserConfig(inscriptis_profile)

    @log
    async def get_index_reqs(self) -> set[Request]:
        # NOTE The documents are served from content.legislation.vic.gov.au, whose robots.txt disallows everything for all user agents (www.legislation.vic.gov.au likewise disallows /sites/default/files/*.doc and *.pdf, and sets Crawl-delay: 2). The legislation's open licence permits reuse but not automated collection, so the scraper refuses to run unless the Office of the Chief Parliamentary Counsel Victoria permits bulk access.
        raise PermissionError('victorian_legislation is disabled: robots.txt at content.legislation.vic.gov.au disallows / '
                              'for all user agents. Obtain bulk access from the Office of the Chief Parliamentary Counsel '
                              'Victoria before re-enabling.')

        # Retrieve the sitemap index and generate requests for every page of the sitemap.
        resp = (await self.get(f'{self._base_url}/sitemap.xml')).text

        return {Request(url) for url in re.findall(r'<loc>([^<]+)</loc>', resp)}

    @log
    async def get_index(self, req: Request) -> set[Entry]:
        # Retrieve the sitemap page.
        resp = (await self.get(req)).text

        # Extract the paths of all in-force documents.
        paths = [url for url in re.findall(r'<loc>([^<]+)</loc>', resp) if '/in-force/' in url]

        # Create entries from the paths.
        entries = await asyncio.gather(*[self._get_entry(path) for path in paths])

        return {entry for entry in entries if entry}

    @log
    async def _get_entry(self, path: str) -> Entry | None:
        # Retrieve the document's landing page.
        resp = (await self.get(path)).text

        # Extract the Nuxt data payload.
        # NOTE It is possible for in-force pages not to embed a Nuxt data payload or any version candidates (see, eg, index-style pages picked up from the sitemap). This is why it is acceptable to return `None`.
        payload = re.search(r'id="__NUXT_DATA__"[^>]*>(.*?)</script>', resp, re.DOTALL)

        if not payload:
            return None

        try:
            data = json.loads(payload.group(1))

        except ValueError:
            return None

        # Identify the current version of the document from the version history embedded in the payload.
        # NOTE The payload is a `devalue`-serialised flat array wherein objects reference the indices of their values, hence candidate versions are resolved by indexing into the array.
        candidates = []

        for element in data:
            if isinstance(element, dict) and 'id' in element and 'version' in element:
                try:
                    uuid = data[element['id']]
                    version = data[element['version']]
                    date = data[element['date']] if 'date' in element else None
                    status = data[element['status']] if 'status' in element else None

                except (IndexError, TypeError):
                    continue

                if isinstance(uuid, str) and self._uuid_pattern.match(uuid) and isinstance(version, str):
                    candidates.append((status == 'In force', version, uuid, date))

        if not candidates:
            return None

        # Prefer the version flagged as being in force, falling back to the highest-numbered version.
        _, version, uuid, date = max(candidates)

        # Determine the document's type from its path.
        type = 'primary_legislation' if '/in-force/acts/' in path else 'secondary_legislation'

        # Extract the document's title from the page's title.
        title = re.search(r'<title>([^<|]+)', resp)
        title = title.group(1).strip() if title else path.rstrip('/').rsplit('/', 1)[-1].replace('-', ' ').title()

        # Preserve the document's public path and version in its version id so that `_get_doc` can reconstruct the document's url.
        slug = path.split('/in-force/', 1)[1].strip('/')

        return Entry(
            request=Request(f'{self._base_url}/api/tide/in-force?id={uuid}'),
            version_id=f'{date}/{slug}/{version}',
            source=self.source,
            type=type,
            jurisdiction=self._jurisdiction,
            date=date,
            title=title,
        )

    @log
    async def _get_doc(self, entry: Entry) -> Document | None:
        # Retrieve the locations of the files constituting the current version of the document.
        resp = (await self.get(entry.request)).json

        # Extract the url of the DOCX version of the document.
        files = resp.get('version') or []
        docx_files = [file for file in files if file.get('extension') == 'docx' and file.get('url')]

        if not docx_files:
            warning(f'Unable to find a DOCX file for {entry.request.path}. Returning `None`.')

            return None

        # Retrieve the DOCX version of the document.
        docx_resp = await self.get(docx_files[0]['url'])

        # Convert the document to HTML.
        try:
            html = docx2html(docx_resp.stream)

        except BadZipFile as e:
            warning(f'Unable to convert DOCX file from {docx_files[0]["url"]} to HTML. BadZipFile encountered: {e}. Returning `None`.')

            return None

        # Extract text from the generated HTML.
        etree = lxml.html.fromstring(html.value)
        text = CustomInscriptis(etree, self._inscriptis_config).get_text()

        # Reconstruct the document's public url from its version id.
        slug_and_version = entry.version_id.split(':', 1)[1].split('/', 1)[1]

        # Return the document.
        return make_doc(
            version_id=entry.version_id,
            type=entry.type,
            jurisdiction=entry.jurisdiction,
            source=entry.source,
            mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            date=entry.date,
            citation=entry.title,
            url=f'{self._base_url}/in-force/{slug_and_version}',
            text=text,
        )
