import re
import asyncio

from datetime import datetime, timedelta
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


class ActLegislation(Scraper):
    """A scraper for the ACT Legislation Register.

    Current acts and subordinate laws are enumerated from the register's browse
    pages, each document's landing page names its current republication, and
    the register serves the current consolidated version of every document as
    a DOCX at a stable, derivable url."""

    def __init__(self,
                 indices_refresh_interval: bool | timedelta = None,
                 index_refresh_interval: bool | timedelta = None,
                 semaphore: asyncio.Semaphore = None,
                 session: aiohttp.ClientSession = None,
                 thread_pool_executor: ThreadPoolExecutor = None,
                 ocr_semaphore: asyncio.Semaphore = None,
                 ) -> None:
        super().__init__(
            source='act_legislation',
            indices_refresh_interval=indices_refresh_interval,
            index_refresh_interval=index_refresh_interval,
            semaphore=semaphore or asyncio.Semaphore(10), # Employ a lower semaphore limit to avoid overloading the register.
            session=session,
            thread_pool_executor=thread_pool_executor,
            ocr_semaphore=ocr_semaphore
        )

        self._jurisdiction = 'australian_capital_territory'
        self._base_url = 'https://www.legislation.act.gov.au'

        # Create a custom Inscriptis CSS profile.
        inscriptis_profile = CSS_PROFILES['strict'].copy()
        inscriptis_profile['p'] = HtmlElement(display=Display.block)
        inscriptis_profile |= dict.fromkeys(('h1', 'h2', 'h3', 'h4', 'h5'), HtmlElement(display=Display.block, margin_before=1))

        self._inscriptis_config = CustomParserConfig(inscriptis_profile)

    @log
    async def get_index_reqs(self) -> set[Request]:
        return {
            Request(f'{self._base_url}/results?category={category}&status=Current&action=browse')

            for category in {'cAct', 'cSub'}
        }

    @log
    async def get_index(self, req: Request) -> set[Entry]:
        # Retrieve the browse page.
        resp = (await self.get(req)).text

        # Extract the paths and titles of all listed documents.
        rows = re.findall(r'<a\s+href="(/(?:a|sl)/[^"#]+)/?"\s*>([^<]+)</a>', resp)
        rows = {(path.rstrip('/'), title.strip()) for path, title in rows}

        # Create entries from the rows.
        entries = await asyncio.gather(*[self._get_entry(path, title) for path, title in rows])

        return {entry for entry in entries if entry}

    @log
    async def _get_entry(self, path: str, title: str) -> Entry | None:
        # Retrieve the document's landing page.
        resp = (await self.get(f'{self._base_url}{path}/')).text

        # Extract the document's current republication number and effective date.
        current = re.search(r'<option value="[^"]*current"\s+selected\s*>R(\d+) \(current\) ([^<]+)<', resp)

        if current:
            republication, date = current.groups()

            try:
                date = datetime.strptime(date.strip(), '%d %B %Y').strftime('%Y-%m-%d')

            except ValueError:
                date = None

        # NOTE It is possible for documents never to have been republished (ie, documents as originally made). Such documents are treated as republication zero and dated from their landing page's notification details where possible.
        else:
            republication = '0'
            date = None

        # Determine the document's type from its path.
        type = 'primary_legislation' if path.startswith('/a/') else 'secondary_legislation'

        # Extract the document's id from its path (eg, `/a/2001-14` becomes `a/2001-14`).
        slug = path.lstrip('/')
        number = slug.split('/', 1)[1]

        return Entry(
            request=Request(f'{self._base_url}/DownloadFile/{slug}/current/DOCX/{number}.DOCX'),
            version_id=f'r{republication}-{date}/{slug}',
            source=self.source,
            type=type,
            jurisdiction=self._jurisdiction,
            date=date,
            title=title,
        )

    @log
    async def _get_doc(self, entry: Entry) -> Document | None:
        # Retrieve the DOCX version of the document.
        resp = await self.get(entry.request)

        # Convert the document to HTML.
        # NOTE It is possible for very old imperial enactments not to be available as DOCX files, in which case the register responds with an HTML error page. This is why it is acceptable to return `None`.
        try:
            html = docx2html(resp.stream)

        except BadZipFile as e:
            warning(f'Unable to convert DOCX file from {entry.request.path} to HTML. BadZipFile encountered: {e}. Returning `None`.')

            return None

        # Extract text from the generated HTML.
        etree = lxml.html.fromstring(html.value)
        text = CustomInscriptis(etree, self._inscriptis_config).get_text()

        # Reconstruct the document's public url from its version id.
        slug = entry.version_id.split(':', 1)[1].split('/', 1)[1]

        # Return the document.
        return make_doc(
            version_id=entry.version_id,
            type=entry.type,
            jurisdiction=entry.jurisdiction,
            source=entry.source,
            mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            date=entry.date,
            citation=entry.title,
            url=f'{self._base_url}/{slug}/',
            text=text,
        )
