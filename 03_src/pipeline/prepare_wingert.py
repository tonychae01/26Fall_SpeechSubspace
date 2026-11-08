"""Download the smallest paper recording and a separate NEMS demo CNN."""
import hashlib
import io
from pathlib import Path
import struct
import urllib.request
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[2]
RECORD = 'https://zenodo.org/records/18331549/files/'
NAME = 'CLT027c_0ac9836c26905c43e7527a70a89c623586b42562.tgz'
SHA256 = 'bb6217157432696107ea4fd4268757c1d04470011d16c24183796843c8c11263'
DEMO = 'https://nemspublic.s3.us-west-2.amazonaws.com/sample_models/CNN-gtgram.fs100.ch32-norm.l1.json'


def download(url, path):
    if path.exists():
        return
    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read()
    path.write_bytes(data)


def main():
    raw = ROOT / '01_raw_data/wingert2026'
    external = ROOT / '06_external/wingert2026'
    raw.mkdir(parents=True, exist_ok=True)
    external.mkdir(parents=True, exist_ok=True)
    path = raw / NAME
    if not path.exists():
        # CLT027c is the first member of this fixed-version ZIP (5 MB vs. 4.6 GB).
        request = urllib.request.Request(RECORD + 'recordings.zip',
                                         headers={'Range': 'bytes=0-5236340'})
        with urllib.request.urlopen(request, timeout=90) as response:
            if response.status != 206:
                raise RuntimeError('Server ignored Range; refusing full-archive download.')
            data = response.read()
        if data[:4] != b'PK\x03\x04':
            raise ValueError('Unexpected ZIP header.')
        header = struct.unpack_from('<4s5H3I2H', data)
        name_len, extra_len = header[-2:]
        if Path(data[30:30+name_len].decode()).name != NAME or header[3] != 8:
            raise ValueError('Recording archive layout changed.')
        start = 30 + name_len + extra_len
        recording = zlib.decompress(data[start:start+5235317], -15)
        if hashlib.sha256(recording).hexdigest() != SHA256:
            raise ValueError('Recording checksum mismatch.')
        path.write_bytes(recording)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Local recording checksum mismatch.')
    for name in ['README.md', 'aud_subspace_fit_demo.ipynb', 'models.zip']:
        download(RECORD + name, external / name)
    download(DEMO, external / 'nems_demo_cnn32.json')
    with zipfile.ZipFile(io.BytesIO((external / 'models.zip').read_bytes())) as archive:
        count = sum(not item.is_dir() for item in archive.infolist())
    print(f'Recording ready: {path}\nPaper model archive: {count} files.')
    print('The NEMS demo CNN is a separate model, not a paper-site checkpoint.')


if __name__ == '__main__':
    main()
