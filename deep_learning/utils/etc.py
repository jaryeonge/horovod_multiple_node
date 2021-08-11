import re
from six import unichr


def convert_seconds(seconds):
    ''' This funciton returns the time converted to day, hours, minutes and seconds. '''
    day = seconds // 86400
    hour = (seconds % 86400) // 3600
    minute = (seconds % 3600) // 60
    second = seconds % 60
    return f'{int(day)}d {int(hour)}h {int(minute)}m {int(second)}s'


chosung_list = (
    u'ㄱ', u'ㄲ', u'ㄴ', u'ㄷ', u'ㄸ', u'ㄹ', u'ㅁ', u'ㅂ', u'ㅃ', u'ㅅ',
    u'ㅆ', u'ㅇ', u'ㅈ', u'ㅉ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ'
)

jungsung_list = (
    u'ㅏ', u'ㅐ', u'ㅑ', u'ㅒ', u'ㅓ', u'ㅔ', u'ㅕ', u'ㅖ', u'ㅗ', u'ㅘ',
    u'ㅙ', u'ㅚ', u'ㅛ', u'ㅜ', u'ㅝ', u'ㅞ', u'ㅟ', u'ㅠ', u'ㅡ', u'ㅢ', u'ㅣ'
)

jongsung_list = (
    u'', u'ㄱ', u'ㄲ', u'ㄳ', u'ㄴ', u'ㄵ', u'ㄶ', u'ㄷ', u'ㄹ', u'ㄺ',
    u'ㄻ', u'ㄼ', u'ㄽ', u'ㄾ', u'ㄿ', u'ㅀ', u'ㅁ', u'ㅂ', u'ㅄ', u'ㅅ',
    u'ㅆ', u'ㅇ', u'ㅈ', u'ㅊ', u'ㅋ', u'ㅌ', u'ㅍ', u'ㅎ'
)

JAMO = chosung_list + jungsung_list + jongsung_list[1:]

NUM_CHO = 19
NUM_JOONG = 21
NUM_JONG = 28

FIRST_HANGUL_UNICODE = 0xAC00  # '가'
LAST_HANGUL_UNICODE = 0xD7A3  # '힣'
#                      AB C DEFGHIJK L M N OPQR ST UVWXYZ
ENG_KOR_SUBSTITUENT = { 'B': 'ㅂ', 'C':'ㄱ', 'K':'ㄱ', 'L':'ㄹ', 'M':'ㅁ', 'N':'ㄴ', 'R':'ㄹ', 'T':'ㅅ'}


def compose_letter(chosung, jungsung, jongsung):
    ''' This function returns a Hangul letter by composing the specified chosung, jungsung, and jongsung.
    @param chosung
    @param jungsung
    @param jongsung the terminal Hangul letter. This is optional if you do not need a jongsung. '''

    if jongsung is None: jongsung = u''

    try:
        chosung_index = chosung_list.index(chosung)
        joongsung_index = jungsung_list.index(jungsung)
        jongsung_index = jongsung_list.index(jongsung)
    except Exception:
        raise ValueError('No valid Hangul character index')

    return unichr(0xAC00 + chosung_index * NUM_JOONG * NUM_JONG + joongsung_index * NUM_JONG + jongsung_index)


def compose_hangul(text):
    ''' This function returns a Hangul sentence by composing the specified chosung, jungsung, and jongsung '''
    text = text.replace('[PAD]', '')
    text = re.sub('\s+', ' ', text).strip()

    result = ''
    CHO = 0
    JUNG = 1
    JONG = 2
    status = CHO
    space = False
    for i in range(len(text)):
        if text[i] == ' ' and status == CHO:
            result += text[i]
        elif text[i] == ' ' and status == JONG:
            space = True
        elif text[i] in chosung_list and status == CHO:
            chosung = text[i]
            status = JUNG
        elif text[i] in jungsung_list and status == JUNG:
            jungsung = text[i]
            status = JONG
            if i == len(text) - 1:
                result += compose_letter(chosung, jungsung, '')
        elif text[i] in jongsung_list and status == JONG:
            if i == len(text) - 1:
                jongsung = text[i]
                status = CHO
                result += compose_letter(chosung, jungsung, jongsung)
            else:
                j = 0
                while True:
                    j += 1
                    if i + j == len(text) - 1:
                        if text[i+j] in jungsung_list:
                            chosung = text[i]
                            jungsung = text[i+j]
                            status = CHO
                            result += compose_letter(chosung, jungsung, '')
                            break
                        else:
                            jongsung = text[i]
                            status = CHO
                            result += compose_letter(chosung, jungsung, jongsung)
                            break
                    else:
                        if text[i+j] in jungsung_list:
                            result += compose_letter(chosung, jungsung, '')
                            if space:
                                result += ' '
                                space = False
                            chosung = text[i]
                            status = JUNG
                            break
                        elif text[i+j] == ' ':
                            space = True
                        else:
                            jongsung = text[i]
                            status = CHO
                            result += compose_letter(chosung, jungsung, jongsung)
                            break

    return result
