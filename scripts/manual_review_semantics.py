"""
Classificação de sentimento a partir apenas do texto (simula anotador humano).

Não utiliza rótulos de modelos — apenas o campo textual. Preferência por NEUTRO
em caso de ambiguidade ou polaridade fraca.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Final, Literal

SentLabel = Literal["POSITIVO", "NEGATIVO", "NEUTRO"]

# Polaridade fraca → NEUTRO (ambiguidade)
SCORE_THRESHOLD: Final[float] = 1.25

_POS_EMOJI = frozenset(
    "😀😃😄😁😊🙂😍🤩😻👍💪🎉✨🔥❤️💚💙🙌👏✅🚀💯⭐🌟"
)
_NEG_EMOJI = frozenset(
    "😠😡🤬😤😭😢😞😒👎💀😰😨🤮❌⚠️😣😖"
)

_POS_WORDS: Final[frozenset[str]] = frozenset(
    """
    adorei amei amor ansioso ansiosa ansiosos ansiosas ansioso ansiosa
    animado animada animados animadas animação animado
    bacana beleza benção bênção bom boa bons boas boazuda boazudo
    brilhante capricho celebração celebrar comemorar confiante confiança
    curti curtindo curte demais divertido divertida empolgado empolgada
    empolgante encantado encantada encantados encantadas entusiasmado
    entusiasmada entusiasmo excelente excelência extraordinário extraordinária
    fantástico fantástica fantásticos feliz felizes felicidade
    fenomenal foda fodao fodástico fodástica genial gigante gloria glorioso
    gostei gostoso gostosa gratidão gratuito incrível incrivel
    legal lindo linda lindos lindas love maravilha maravilhoso maravilhosa
    massa melhor melhores milagre milagroso milagrosa motivado motivada
    nota notável ótimo otimo ótima otima ótimos otimos perfeito perfeita
    positivo positiva positivos positivas prazer radiante recomendo
    satisfação satisfeito satisfeita sensacional show sucesso
    top vencedor vencedora vitória vitoria viva adore
    parabéns parabens
    evolução evolucao avanço avanços avanco avancos inovador inovadora
    otimismo otimista esperança esperanca futuro promissor
    """.split()
)

_NEG_WORDS: Final[frozenset[str]] = frozenset(
    """
    absurdo absurda absurdos absurdas medo pavor terror horror
    agonia agonizando angustia angústia angustiado angustiada
    assustador assustadora assustado assustada assusta
    atrocidade ruim ruins péssimo pessimo péssima pessima
    pior piores problema problemas preocupado preocupada preocupação
    preocupacao frustrado frustrada frustração frustracao
    decepção decepcao decepcionado decepcionada decepcionante
    triste tristes tristeza raiva irritado irritada irritante
    irritação irritacao nojo nojento nojenta odio ódio odiar
    detesto detestamos detestável detestavel horroroso horrorosa
    lixo merda porcaria bosta cagada desgraça desgraca
    falha falhas falhou bugado bugada quebrado quebrada
    crítica critica criticar criticando reclama reclamar reclamando
    reclamação reclamacao injusto injusta injustiça injustica
    abusivo abusiva abuso exploração exploracao perigoso perigosa
    ameaça ameaça ameaca ameaçador ameaçadora
    censura censurar proíbe proibe proibido proibida
    desemprego demissão demissao demitido demitida
    """.split()
)

_NEGATION: Final[frozenset[str]] = frozenset(
    "não nao nunca jamais nem sem nada de pouco".split()
)

_INTENS_POS: Final[frozenset[str]] = frozenset(
    "muito demais super ultra mega hiper super hiper".split()
)
_INTENS_NEG: Final[frozenset[str]] = frozenset(
    "muito demais super ultra mega hiper péssimo pessimo horrível horrivel".split()
)


def _strip_accents(s: str) -> str:
    n = unicodedata.normalize("NFD", s)
    return "".join(c for c in n if unicodedata.category(c) != "Mn")


def _tokens(text: str) -> list[str]:
    t = text.lower()
    t = _strip_accents(t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#(\w+)", r" \1 ", t)
    return re.findall(r"[a-záéíóúâêôãõç]+", t, flags=re.I)


def _emoji_hits(text: str) -> tuple[int, int]:
    pos = sum(1 for ch in text if ch in _POS_EMOJI)
    neg = sum(1 for ch in text if ch in _NEG_EMOJI)
    return pos, neg


def _label_from_emoji_only(raw: str) -> SentLabel | None:
    ep, en = _emoji_hits(raw)
    if ep > en + 1:
        return "POSITIVO"
    if en > ep + 1:
        return "NEGATIVO"
    return None


def _word_delta(i: int, w: str, toks: list[str], flip: float) -> float:
    if w in _POS_WORDS:
        boost = 1.35
        if i > 0 and toks[i - 1] in _INTENS_POS:
            boost *= 1.2
        return flip * boost
    if w in _NEG_WORDS:
        boost = 1.35
        if i > 0 and toks[i - 1] in _INTENS_NEG:
            boost *= 1.2
        return flip * (-boost)
    return 0.0


def _lexicon_score(toks: list[str]) -> float:
    score = 0.0
    negation_window = 0
    for i, w in enumerate(toks):
        if w in _NEGATION:
            negation_window = 4
            continue
        flip = -1.0 if negation_window > 0 else 1.0
        if negation_window > 0:
            negation_window -= 1
        score += _word_delta(i, w, toks, flip)
    return score


def classify_from_content_only(text: str | None) -> SentLabel:
    """
    Rótulo único POSITIVO | NEGATIVO | NEUTRO com base só no texto.
    Polaridade fraca ou mista → NEUTRO.
    """
    if not text or not str(text).strip():
        return "NEUTRO"

    raw = str(text)
    toks = _tokens(raw)
    if not toks:
        return _label_from_emoji_only(raw) or "NEUTRO"

    ep, en = _emoji_hits(raw)
    score = 1.4 * ep - 1.4 * en + _lexicon_score(toks)

    if score > SCORE_THRESHOLD:
        return "POSITIVO"
    if score < -SCORE_THRESHOLD:
        return "NEGATIVO"
    return "NEUTRO"
