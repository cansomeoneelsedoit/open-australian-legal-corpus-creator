import re
import asyncio

from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import lxml.html

from pdfminer.high_level import extract_text
from inscriptis.css_profiles import CSS_PROFILES
from inscriptis.model.html_element import HtmlElement
from inscriptis.html_properties import Display

from ..data import Entry, Request, Document, make_doc
from ..helpers import log, warning
from ..scraper import Scraper
from ..custom_inscriptis import CustomInscriptis, CustomParserConfig


class QueenslandJudgments(Scraper):
    """A scraper for the Queensland Judgments database of the Supreme Court Library Queensland.

    The database's entire collection is enumerated through its DataTables
    endpoint, which requires a session cookie and CSRF token obtained from any
    case law page. Judgment text is extracted from the enhanced HTML view,
    falling back to the embedded text layers of originals served as PDFs."""

    def __init__(self,
                 indices_refresh_interval: bool | timedelta = None,
                 index_refresh_interval: bool | timedelta = None,
                 semaphore: asyncio.Semaphore = None,
                 session: aiohttp.ClientSession = None,
                 thread_pool_executor: ThreadPoolExecutor = None,
                 ocr_semaphore: asyncio.Semaphore = None,
                 ) -> None:
        super().__init__(
            source='queensland_judgments',
            indices_refresh_interval=indices_refresh_interval,
            index_refresh_interval=index_refresh_interval,
            semaphore=semaphore or asyncio.Semaphore(5), # Employ a low semaphore limit to avoid overloading the database.
            session=session,
            thread_pool_executor=thread_pool_executor,
            ocr_semaphore=ocr_semaphore
        )

        self._jurisdiction = 'queensland'
        self._type = 'decision'
        self._base_url = 'https://www.queenslandjudgments.com.au'
        self._page_length = 100

        # NOTE The database serves a stub page to clients without a browser user agent.
        self._ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'

        self._csrf: str | None = None
        self._csrf_lock = asyncio.Lock()

        # Create a custom Inscriptis CSS profile.
        inscriptis_profile = CSS_PROFILES['strict'].copy()
        inscriptis_profile['p'] = HtmlElement(display=Display.block)
        inscriptis_profile |= dict.fromkeys(('h1', 'h2', 'h3', 'h4', 'h5'), HtmlElement(display=Display.block, margin_before=1))

        self._inscriptis_config = CustomParserConfig(inscriptis_profile)

    async def _get_csrf(self, refresh: bool = False) -> str:
        """Retrieve a CSRF token (and accompanying session cookie) for the DataTables endpoint."""

        async with self._csrf_lock:
            if self._csrf is None or refresh:
                resp = (await self.get(Request(f'{self._base_url}/caselaw/qsc', headers={'User-Agent': self._ua}))).text
                self._csrf = re.search(r'<meta name="csrf-token" content="([^"]+)"', resp).group(1)

            return self._csrf

    async def _datatable(self, start: int) -> dict:
        """Retrieve a page of the collection from the DataTables endpoint."""

        csrf = await self._get_csrf()

        for attempt in range(2):
            resp = await self.get(Request(
                f'{self._base_url}/caselaw/datatable-case-law',
                method='post',
                data={'start': str(start), 'length': str(self._page_length), '_csrf': csrf},
                headers={'X-CSRF-TOKEN': csrf, 'X-Requested-With': 'XMLHttpRequest', 'User-Agent': self._ua},
            ))

            try:
                return resp.json

            # NOTE CSRF tokens are session-bound and expire, in which case the endpoint redirects to an HTML page. Refreshing the token and retrying recovers from this.
            except ValueError:
                csrf = await self._get_csrf(refresh=True)

        raise ValueError(f'Unable to retrieve page {start} of the Queensland Judgments collection.')

    @log
    async def get_index_reqs(self) -> set[Request]:
        # NOTE The database's robots.txt disallows, for all user agents, both /caselaw/datatable-case-law (the only listing of the collection, used below) and /*pdf (the fallback in `_get_doc`), and the database began refusing this scraper with HTTP 403 on 23 October 2026. The scraper refuses to run unless the Supreme Court Library Queensland grants bulk access.
        raise PermissionError('queensland_judgments is disabled: robots.txt at queenslandjudgments.com.au disallows '
                              '/caselaw/datatable-case-law and /*pdf. Obtain bulk access from the Supreme Court Library '
                              'Queensland before re-enabling.')

        # Determine the total number of records in the collection.
        total = (await self._datatable(0))['recordsTotal']

        # Generate requests for every page of the collection.
        # NOTE The page offset is encoded in a query string. The requests are not retrieved directly but instead identify the pages to be retrieved by `get_index` through the DataTables endpoint, which requires a fresh CSRF token.
        return {
            Request(f'{self._base_url}/caselaw/datatable-case-law?start={start}')

            for start in range(0, int(total), self._page_length)
        }

    @log
    async def get_index(self, req: Request) -> set[Entry]:
        # Extract the page offset from the request.
        start = int(req.path.rsplit('=', 1)[1])

        # Retrieve the page.
        resp = await self._datatable(start)

        entries = set()

        for row in resp.get('data', []):
            # Skip unavailable judgments and articles.
            if not row.get('tbl_case_id') or row.get('tbl_case_is_article') == '1':
                continue

            # NOTE `tbl_case_cite_url` takes the form `<year>-<abbreviation slug>` (eg, `1879-qlj-supp`) and provides the slugified court abbreviation used by judgment urls, which cannot reliably be derived from the raw abbreviation (eg, `QLJ (Supp)`).
            cite_url = row.get('tbl_case_cite_url') or ''
            abbrev = cite_url.split('-', 1)[1] if '-' in cite_url else (row.get('tbl_case_cite_abbrev_raw') or '').lower()
            year = row.get('tbl_case_cite_year')
            number = row.get('tbl_case_cite_no')

            if not all((abbrev, year, number)):
                continue

            date = row.get('tbl_case_date_delivered') or None
            version = row.get('tbl_case_publication_date') or date or 'unknown'
            name = row.get('tbl_case_name') or ''
            citation = row.get('tbl_case_cite_full') or f'[{year}] {abbrev.upper()} {number}'

            entries.add(Entry(
                request=Request(f'{self._base_url}/caselaw/{abbrev}/{year}/{number}', headers={'User-Agent': self._ua}),
                version_id=f'{version}/{abbrev}-{year}-{number}',
                source=self.source,
                type=self._type,
                jurisdiction=self._jurisdiction,
                date=date,
                title=f'{name} {citation}'.strip(),
            ))

        return entries

    @log
    async def _get_doc(self, entry: Entry) -> Document | None:
        # Retrieve the enhanced view of the judgment.
        resp = await self.get(entry.request)

        text = ''
        mime = 'text/html'

        try:
            etree = lxml.html.fromstring(resp.text)
            wrapper = etree.xpath('//div[@id="caseview_html_footnotes_wrapper"]')

            if wrapper:
                text = CustomInscriptis(wrapper[0], self._inscriptis_config).get_text()

        except Exception as e:
            warning(f'Unable to parse the enhanced view of {entry.request.path} ({type(e).__name__}: {e}). Falling back to the PDF original.')

        # Fall back to the original PDF for judgments without an enhanced view (eg, backset judgments).
        if len(re.sub(r'\W', '', text)) < 9:
            pdf_resp = await self.get(Request(f'{entry.request.path}/pdf', headers={'User-Agent': self._ua}))

            try:
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(self.thread_pool_executor, extract_text, pdf_resp.stream)
                mime = 'application/pdf'

            except Exception as e:
                warning(f'Unable to extract text from the PDF original of {entry.request.path} ({type(e).__name__}: {e}). Returning `None`.')

                return None

        # Return the document.
        return make_doc(
            version_id=entry.version_id,
            type=entry.type,
            jurisdiction=entry.jurisdiction,
            source=entry.source,
            mime=mime,
            date=entry.date,
            citation=entry.title,
            url=entry.request.path,
            text=text,
        )
