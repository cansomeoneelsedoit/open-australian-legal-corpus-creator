import re
import asyncio

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import aiohttp

from pdfminer.high_level import extract_text

from ..ocr import pdf2txt
from ..data import Entry, Request, Document, make_doc
from ..helpers import log, warning
from ..scraper import Scraper


class NtLegislation(Scraper):
    """A scraper for the Northern Territory Legislation database.

    In-force acts and subordinate legislation are enumerated from the
    database's by-title listings, each document's landing page names its
    current reprint, and the Sitecore API serves the current reprint of every
    document as a PDF. Text is extracted from the PDFs' embedded text layers,
    falling back to OCR for documents without one."""

    def __init__(self,
                 indices_refresh_interval: bool | timedelta = None,
                 index_refresh_interval: bool | timedelta = None,
                 semaphore: asyncio.Semaphore = None,
                 session: aiohttp.ClientSession = None,
                 thread_pool_executor: ThreadPoolExecutor = None,
                 ocr_semaphore: asyncio.Semaphore = None,
                 ) -> None:
        super().__init__(
            source='nt_legislation',
            indices_refresh_interval=indices_refresh_interval,
            index_refresh_interval=index_refresh_interval,
            semaphore=semaphore or asyncio.Semaphore(10), # Employ a lower semaphore limit to avoid overloading the database.
            session=session,
            thread_pool_executor=thread_pool_executor,
            ocr_semaphore=ocr_semaphore
        )

        self._jurisdiction = 'northern_territory'
        self._base_url = 'https://legislation.nt.gov.au'

    @log
    async def get_index_reqs(self) -> set[Request]:
        return {
            Request(f'{self._base_url}/en/LegislationPortal/Acts/By-Title?status=inforce'),
            Request(f'{self._base_url}/en/LegislationPortal/Subordinate-Legislation/By-Title?status=inforce'),
        }

    @log
    async def get_index(self, req: Request) -> set[Entry]:
        # Determine the document type of the index.
        type = 'primary_legislation' if '/Acts/' in req.path else 'secondary_legislation'

        # Retrieve the index.
        resp = (await self.get(req)).text

        # Extract the paths of all listed documents.
        # NOTE The listing proper lies inside the page's accordions, whereas the site-wide navigation also links a handful of documents. Restricting extraction to paths appearing after the first accordion header keeps the index to the listing.
        body = resp.split('header_', 1)[-1]
        paths = sorted({path for path in re.findall(r'href="(/en/Legislation/[^"#?]+)"', body)})

        # Create entries from the paths.
        entries = await asyncio.gather(*[self._get_entry(path, type) for path in paths])

        return {entry for entry in entries if entry}

    @log
    async def _get_entry(self, path: str, type: str) -> Entry | None:
        # Retrieve the document's landing page.
        resp = (await self.get(f'{self._base_url}{path}')).text

        # Skip documents that are not in force.
        status = re.search(r'<label>Status:</label>\s*<span>([^<]*)</span>', resp)

        if status and status.group(1).strip().lower() != 'in force':
            return None

        # Extract the id of the document's PDF.
        # NOTE It is possible for documents not to have a PDF (see, eg, documents whose landing pages exist but whose reprints have been withdrawn). This is why it is acceptable to return `None`.
        pdf_id = re.search(r'api/sitecore/Act/PDF\?id=(\d+)', resp)

        if not pdf_id:
            return None

        pdf_id = pdf_id.group(1)

        # Extract the document's current reprint, which serves as its version marker.
        reprint = re.search(r'<label>Reprint:</label>\s*<span>([^<]*)</span>', resp)
        reprint = (reprint.group(1).strip() if reprint else '') or 'orig'

        # Extract the document's title.
        slug = path.rsplit('/', 1)[-1]
        title = re.search(r'<h1[^>]*>([^<]+)</h1>', resp)
        title = title.group(1).strip() if title else slug.replace('-', ' ').title()

        return Entry(
            request=Request(f'{self._base_url}/api/sitecore/Act/PDF?id={pdf_id}'),
            version_id=f'{reprint}-{pdf_id}/{slug.lower()}',
            source=self.source,
            type=type,
            jurisdiction=self._jurisdiction,
            date=None,
            title=title,
        )

    @log
    async def _get_doc(self, entry: Entry) -> Document | None:
        # Retrieve the PDF version of the document.
        resp = await self.get(entry.request)

        # Extract text from the PDF's embedded text layer.
        try:
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(self.thread_pool_executor, extract_text, resp.stream)

        except Exception as e:
            warning(f'Unable to extract embedded text from {entry.request.path} ({type(e).__name__}: {e}). Falling back to OCR.')
            text = ''

        # Fall back to OCR for documents without a usable text layer.
        if len(re.sub(r'\W', '', text)) < 9:
            text = await pdf2txt(resp.stream, self.ocr_batch_size, self.thread_pool_executor, self.ocr_semaphore, scale=2)

        # Attempt to extract the date the document is in force as at.
        date = entry.date

        if not date and (as_at := re.search(r'As in force at (\d{1,2} [A-Z][a-z]+ \d{4})', text)):
            try:
                date = datetime.strptime(as_at.group(1), '%d %B %Y').strftime('%Y-%m-%d')

            except ValueError:
                date = None

        # Reconstruct the document's public url from its version id.
        slug = entry.version_id.split('/', 1)[1]

        # Return the document.
        return make_doc(
            version_id=entry.version_id,
            type=entry.type,
            jurisdiction=entry.jurisdiction,
            source=entry.source,
            mime='application/pdf',
            date=date,
            citation=entry.title,
            url=f'{self._base_url}/en/Legislation/{slug.upper()}',
            text=text,
        )
