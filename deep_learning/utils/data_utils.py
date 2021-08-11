import paramiko
import os
import re
import soundfile as sf
import librosa

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan, bulk
from torch.utils.data import IterableDataset, Dataset


class ESAPI:
    """
    Class for connection with elasticsearch.
    This require config class.

    Config attributes
    -----------------
    host: str, elasticsearch host
    port: int, elasticsearch port
    username: str, elasticsearch user
    password: str, elasticsearch password
    data_path: str, elasticsearch index
    query: dict, search query
    """
    def __init__(self, config):
        self.config = config
        self.es = Elasticsearch(**{'host': config.host,
                                   'port': config.port,
                                   'http_auth': (config.username, config.password),
                                   'timeout': 1000000})

    def scan_gen(self):
        return scan(self.es, index=self.config.data_path, query=self.config.query, scroll='1d')

    def counting(self):
        return self.es.count(index=self.config.data_path, body=self.config.query)

    def bulk_gen(self, fn: callable):
        bulk(self.es, fn)


class ESDataset(IterableDataset):
    def __init__(self, config):
        self.es = ESAPI(config)

    def gen_iterator(self, scan_fn):
        for res in scan_fn:
            data = res['_source']
            yield data

    def __iter__(self):
        scan_fn = self.es.scan_gen()
        gen = self.gen_iterator(scan_fn)
        return iter(gen)


def eng_to_kor(text):
    eng_dic = {'A': '에이',
               'B': '비',
               'C': '씨',
               'D': '디',
               'E': '이',
               'F': '에프',
               'G': '지',
               'H': '에이치',
               'I': '아이',
               'J': '제이',
               'K': '케이',
               'L': '엘',
               'M': '엠',
               'N': '엔',
               'O': '오',
               'P': '피',
               'Q': '큐',
               'R': '알',
               'S': '에스',
               'T': '티',
               'U': '유',
               'V': '브이',
               'W': '더블유',
               'X': '엑스',
               'Y': '와이',
               'Z': '제트'}
    measure_dic = {'kg': '킬로그램',
                   'km': '킬로미터',
                   'mm': '미리미터'}

    for k, v in measure_dic.items():
        text = text.replace(k, v)

    for k, v in eng_dic.items():
        text = text.replace(k, v)
        text = text.replace(k.lower(), v)

    return text


def uncased_convert(text):
    # 유니코드 한글, 시작: 44032, 끝: 55199
    base_code, chosung, jungsung = 44032, 588, 28

    # 초성 리스트. 00 ~ 18
    chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    # 중성 리스트. 00 ~ 20
    jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                     'ㅣ']

    # 종성 리스트. 00 ~ 27 + 1(1개 없음)
    jongsung_list = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    split_keyword_list = list(text)

    result = ''
    for keyword in split_keyword_list:
        # 한글 여부 check 후 분리
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
            char_code = ord(keyword) - base_code
            char1 = int(char_code / chosung)
            result += str(chosung_list[char1])

            char2 = int((char_code - (chosung * char1)) / jungsung)
            result += str(jungsung_list[char2])

            char3 = int((char_code - (chosung * char1) - (jungsung * char2)))
            if char3 == 0:
                pass
            else:
                result += str(jongsung_list[char3])
        else:
            result += '|'

    return result


class SftpDataset(Dataset):
    ''' This class loads the data from remote server with sftp '''
    def __init__(self, config):
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_client.connect(hostname=config.host,
                                port=config.port,
                                username=config.username,
                                password=config.password)
        self.sftp_client = self.ssh_client.open_sftp()
        self.data_path = config.data_path
        self.data_list = [os.path.join(self.data_path, p) for p in self.sftp_client.listdir(self.data_path)]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        wav_path = (os.path.join(self.data_list[idx], 'voice.wav'))
        txt_path = (os.path.join(self.data_list[idx], 'label.txt'))

        input_values = self.preprocessing_wav(wav_path)
        labels = self.preprocessing_txt(txt_path)
        return {'input_values': input_values, 'labels': labels}

    def preprocessing_txt(self, path):
        with self.sftp_client.open(path, encoding='cp949') as file:
            line = file.readline()
            txt_info = line.replace('\n', '')

            r_txt = txt_info.replace('o/', '').replace('l/', '').replace('b/', '').replace('n/', '').replace('u/', '').replace('+', '').replace('*', '')
            r_txt = re.sub('\([가-힣0-9a-zA-Z\s\.\%\:\,\?\!]+\)\/', '', r_txt)
            r_txt = r_txt.replace('(', '').replace(')', '').replace('/', '').replace('?', '').replace('!', '').replace(
                '.', '').replace(',', '').replace('-', '').replace('%', '퍼센트')
            r_txt = re.sub('\s+', ' ', r_txt).strip()
            r_txt = eng_to_kor(r_txt)
            r_txt = uncased_convert(r_txt)
        return r_txt

    def preprocessing_wav(self, path):
        speech, fs = sf.read(self.sftp_client.open(path))
        if len(speech.shape) > 1:
            speech = speech[:, 0] + speech[:, 1]
        if fs != 16000:
            speech = librosa.resample(speech, fs, 16000)
        return speech
