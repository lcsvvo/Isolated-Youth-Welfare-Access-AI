"""Narrow BM25 alias for a confirmed wish for a place to connect with people."""
import re

PLACE_WISH = re.compile(r'(?:외로|공허).{0,30}(?:어디.{0,10}갈|갈\s*만한\s*곳|만날\s*곳)'
                        r'.{0,22}(?:좋겠|찾|있으면|알려|없을까)')
DECLINED = re.compile(r'원하지\s*않|싶지\s*않|필요\s*없|혼자.{0,8}(?:좋|편)')


def communication_tokens(tokens, user_text, domains, tokenize):
    """Add 소통 only for the explicit wish and a confirmed relationship need."""
    if ('관계참여' not in domains or not PLACE_WISH.search(user_text)
            or DECLINED.search(user_text)):
        return []
    return [token for token in tokenize('소통') if token not in tokens]
