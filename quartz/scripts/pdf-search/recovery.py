"""Require native PDF evidence before recovering an empty marginal illustration."""

from pathlib import Path
import hashlib
import re

import pymupdf


def reviewed_region_text(
  source: Path, number: int, bounds: list[float], content: str | None
) -> dict[str, str] | None:
  if number == 312 and bounds in (
    [827, 409, 890, 414],
    [383, 569, 434, 575],
    [731, 639, 767, 644],
    [643, 748, 696, 754],
    [727, 792, 762, 798],
    [788, 869, 868, 879],
  ):
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
      source_hash
      != 'dfc364b1cdc0e71f18a2f6a5bf4b4b5660fb4de2f93a155e4cae949f3e9ae991'
    ):
      return None
    return {
      'content': '[unreadable: miniature artwork caption; literal glyph recognition unresolved at source resolution]',
      'recovery': 'reviewed-region-text-v1',
      'review_status': 'reviewed-partial',
      'review_recognition': 'approved-transcription-before-ocr',
      'review_source_sha256': source_hash,
      'review_physical_page': '312',
      'review_bounds': str(bounds),
      'review_models': 'gemini-4-argon-high',
      'review_approved_by': 'Codex image review under aarnphm authorization',
      'review_date': '2026-11-04',
      'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md#ia2001-page-312',
      'review_approval': 'Standing marked-partial specimen recovery authorization, 2026-11-03; monitor recovery instruction, 2026-11-04',
    }
  if number == 104 and bounds in ([113, 584, 182, 605], [278, 605, 348, 627]):
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
      source_hash
      != 'cf49bbbb18ec9a1928d32530e9676b16c34d696394579b177ab38dbadc368534'
    ):
      return None
    return {
      'content': '[unreadable: miniature printed specimen; literal glyph recognition unresolved at source resolution]',
      'recovery': 'reviewed-region-text-v1',
      'review_status': 'reviewed-partial',
      'review_recognition': 'approved-transcription-before-ocr',
      'review_source_sha256': source_hash,
      'review_physical_page': '104',
      'review_bounds': str(bounds),
      'review_models': 'gemini-4-argon-high',
      'review_approved_by': 'Codex image review under aarnphm authorization',
      'review_date': '2026-11-04',
      'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md#grid-systems-page-104',
      'review_approval': 'Standing marked-partial specimen recovery authorization, 2026-11-03; monitor recovery instruction, 2026-11-04',
    }
  if number == 100 and bounds == [88, 402, 211, 570]:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
      source_hash
      != 'cf49bbbb18ec9a1928d32530e9676b16c34d696394579b177ab38dbadc368534'
    ):
      return None
    return {
      'content': '[unreadable: miniature printed specimen; literal glyph recognition unresolved at source resolution]',
      'recovery': 'reviewed-region-text-v1',
      'review_status': 'reviewed-partial',
      'review_recognition': 'approved-transcription-before-ocr',
      'review_source_sha256': source_hash,
      'review_physical_page': '100',
      'review_bounds': str(bounds),
      'review_models': 'gemini-4-argon-high',
      'review_approved_by': 'Codex image review under aarnphm authorization',
      'review_date': '2026-11-04',
      'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md#grid-systems-page-100',
      'review_approval': 'Standing marked-partial specimen recovery authorization, 2026-11-03; monitor recovery instruction, 2026-11-04',
    }
  if number == 119 and bounds == [198, 491, 355, 523]:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
      source_hash
      != '728e6d0ca3809d504f12f3302ef5303238a7324cf2824b303d64781d597dba7d'
    ):
      return None
    return {
      'content': 'LOCK-STITCH SEWING-MACHINES,\n[unreadable] [unreadable] [unreadable] W[unreadable] [unreadable] WIL[unreadable]N MANUFA[unreadable]-\nTURIN[unreadable] C[unreadable]MPANY, [unreadable] [unreadable] [unreadable]. C[unreadable] C[unreadable]-\n[unreadable], [unreadable] [unreadable] H[unreadable], B[unreadable], C[unreadable], [unreadable]. O[unreadable] [unreadable] [unreadable]-\n[unreadable], 1[unreadable], R[unreadable]-[unreadable], W., L[unreadable] ([unreadable] [unreadable], O[unreadable]-[unreadable]).\nI[unreadable] [unreadable] [unreadable] [unreadable] [unreadable]. I[unreadable] P[unreadable], [unreadable]\nT[unreadable], [unreadable] [unreadable] [unreadable]-[unreadable].',
      'recovery': 'reviewed-region-text-v1',
      'review_status': 'reviewed-partial',
      'review_recognition': 'approved-transcription-before-ocr',
      'review_source_sha256': source_hash,
      'review_physical_page': '119',
      'review_bounds': str(bounds),
      'review_models': 'gemini-4-argon-high',
      'review_approved_by': 'Codex image review under aarnphm authorization',
      'review_date': '2026-11-04',
      'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md#thinking-with-type-page-119',
      'review_approval': 'Standing marked-partial specimen recovery authorization, 2026-11-03; monitor recovery instruction, 2026-11-04',
    }
  if number == 117 and bounds == [275, 108, 368, 183]:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
      source_hash
      != '728e6d0ca3809d504f12f3302ef5303238a7324cf2824b303d64781d597dba7d'
    ):
      return None
    return {
      'content': '[unreadable: ornamental woodcut initial; overlaps region 4]',
      'recovery': 'reviewed-region-text-v1',
      'review_status': 'reviewed-partial',
      'review_recognition': 'approved-transcription-before-ocr',
      'review_source_sha256': source_hash,
      'review_physical_page': '117',
      'review_bounds': str(bounds),
      'review_models': 'gemini-4-argon-high',
      'review_approved_by': 'Codex image review under aarnphm authorization',
      'review_date': '2026-11-04',
      'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md#thinking-with-type-page-117',
      'review_approval': 'Standing marked-partial specimen recovery authorization, 2026-11-03; monitor recovery instruction, 2026-11-04',
    }
  if number == 117 and bounds == [265, 108, 582, 664]:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
      source_hash
      != '728e6d0ca3809d504f12f3302ef5303238a7324cf2824b303d64781d597dba7d'
    ):
      return None
    return {
      'content': "[unreadable: marginal numeral] [unreadable: woodcut initial overlapping region-3]Ν [unreadable: αρ]χη εποιησεν ο θεος [unreadable: το]ν [unreadable: ου]ρανον [unreadable: και] [unreadable: τη]ν γην. [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] [unreadable: inline numeral] η [unreadable: δε] γη ην αο[unreadable: ρατος] [unreadable: και] ακα[unreadable: τα]σκευασ[unreadable: τος]· [unreadable: και] [unreadable: polytonic diacritics]\nσκο[unreadable: το]ς επανω [unreadable: της] α[unreadable: βυσσου]. [unreadable: και] πνευμα [unreadable: θεου] επε[unreadable: φε] [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] ρε[unreadable: το] επανω [unreadable: του] υδα[unreadable: το]ς. [unreadable: inline numeral] [unreadable: και] ειπεν ο θεος, [unreadable: γενη]θητω [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] φως· [unreadable: και] εγενε[unreadable: το] φως. [unreadable: inline numeral] [unreadable: και] ειδεν ο θεος [unreadable: το] φως, οτι [unreadable: polytonic diacritics]\nκαλον. [unreadable: και] διεχω[unreadable: ρι]σεν ο θεος αναμε[unreadable: σο]ν [unreadable: του] φω[unreadable: το]ς, [unreadable: και] αναμε[unreadable: σο]ν [unreadable: του] [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] σκο[unreadable: του]ς. [unreadable: inline numeral] [unreadable: και] εκαλεσεν ο θεος [unreadable: το] φως ημε[unreadable: ρα]ν, [unreadable: και] [unreadable: το] σκο[unreadable: το]ς εκα- [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] λεσε νυκ[unreadable: τα]. [unreadable: και] ε[unreadable: γε]νε[unreadable: το] ε[unreadable: σπε]ρα, [unreadable: και] εγενε[unreadable: το] π[unreadable: ρω]ι, ημερα μια. [unreadable: inline numeral] [unreadable: και] ει- [unreadable: polytonic diacritics]\nπεν ο θεος, Γενηθητω [unreadable: στε]ρεωμα εν μεσω [unreadable: του] υ[unreadable: δατος]· [unreadable: και] ε[unreadable: στω] διαχωριζον [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] αναμε[unreadable: σο]ν υδα[unreadable: το]ς [unreadable: και] υδα[unreadable: το]ς. [unreadable: inline numeral] [unreadable: και] εποιησεν ο θεος [unreadable: το] [unreadable: στε]ρεωμα. [unreadable: και] διε [unreadable: polytonic diacritics]\nχω[unreadable: ρι]σεν ο θεος αναμε[unreadable: σο]ν [unreadable: του] υδα[unreadable: το]ς, ο ην υποκατω [unreadable: του] [unreadable: στε]ρεωμα[unreadable: το]ς, [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] [unreadable: και] αναμε[unreadable: σο]ν [unreadable: του] υδα[unreadable: το]ς, [unreadable: του] επανω [unreadable: του] [unreadable: στε]ρεωμα[unreadable: το]ς. [unreadable: inline numeral] [unreadable: και] εκαλεσεν ο [unreadable: polytonic diacritics]\nθεος [unreadable: το] [unreadable: στε]ρεωμα [unreadable: ου]ρανον. [unreadable: και] ειδεν ο θεος, ο[unreadable: τι] καλον. [unreadable: και] εγενε[unreadable: το] ε[unreadable: σπε] [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] [unreadable: ρα], [unreadable: και] εγενε[unreadable: το] π[unreadable: ρω]ι, ημερα δ[unreadable: ευτε]ρα. [unreadable: inline numeral] [unreadable: και] ειπεν ο θεος, Συναχθητω [unreadable: polytonic diacritics]\n[unreadable: το] υδωρ [unreadable: το] υποκατω [unreadable: του] [unreadable: ουραν]ου εις συνα[unreadable: γω]γην μιαν, [unreadable: και] ο[unreadable: φθη]τω η [unreadable: polytonic diacritics]\nξηρα· [unreadable: και] εγενε[unreadable: το] [unreadable: ου]τως· [unreadable: και] συνηχθη [unreadable: το] υ[unreadable: δω]ρ [unreadable: το] υποκατω [unreadable: του] [unreadable: ουραν]ου [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] εις τας συνα[unreadable: γωγ]ας αυ[unreadable: του], [unreadable: και] ωφθη η ξηρα. [unreadable: inline numeral] [unreadable: και] εκαλεσεν ο θεος [unreadable: τη]ν [unreadable: polytonic diacritics]\nξηραν, γην· [unreadable: και] [unreadable: τα] συ[unreadable: στη]μα[unreadable: τα] [unreadable: τω]ν υ[unreadable: δατ]ων εκαλεσε θαλα[unreadable: σσ]ας. [unreadable: και] ει [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] δεν ο θεος, ο[unreadable: τι] καλον· [unreadable: inline numeral] [unreadable: και] ειπεν ο θεος, Βλα[unreadable: στη]σατω η γη βο[unreadable: τανην] [unreadable: polytonic diacritics]\nχορ[unreadable: του] [unreadable: σπειρο]ν [unreadable: σπ]ερμα [unreadable: κατα] γεν[unreadable: ος] [unreadable: και] καθ' ομοιοτη[unreadable: τα], [unreadable: και] ξυλον [unreadable: polytonic diacritics]\nκαρπιμον ποι[unreadable: ου]ν καρπον· [unreadable: ου] [unreadable: το] [unreadable: σπ]ερμα αυ[unreadable: του] εν αυτω [unreadable: κατα] γεν[unreadable: ος] [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] [unreadable: επι] [unreadable: της] γης· [unreadable: και] εγενε[unreadable: το] [unreadable: ου]τως. [unreadable: inline numeral] [unreadable: και] [unreadable: εξην]εγκεν η γη βο[unreadable: τανην] χορ[unreadable: του] [unreadable: polytonic diacritics]\n[unreadable: σπειρο]ν [unreadable: σπ]ερμα [unreadable: κατα] γεν[unreadable: ος] [unreadable: και] καθ' ομοιοτη[unreadable: τα], [unreadable: και] ξυλον καρπι- [unreadable: polytonic diacritics]\nμον ποι[unreadable: ου]ν καρπον· [unreadable: ου] [unreadable: το] [unreadable: σπ]ερμα αυ[unreadable: του] εν αυτω [unreadable: κατα] γεν[unreadable: ος] [unreadable: επι] [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] [unreadable: της] γης· [unreadable: και] ειδεν ο θεος ο[unreadable: τι] καλον. [unreadable: inline numeral] [unreadable: και] εγενε[unreadable: το] ε[unreadable: σπερα] [unreadable: και] εγενε[unreadable: το] [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] π[unreadable: ρω]ι, ημε[unreadable: ρα] [unreadable: τρι]τη· [unreadable: inline numeral] [unreadable: και] ειπεν ο θεος, [unreadable: γενηθη]τωσ[unreadable: αν] φω[unreadable: στ]ηρες εν τω [unreadable: polytonic diacritics]\n[unreadable: στε]ρεωμα[unreadable: τι] [unreadable: του] [unreadable: ουρανου] ως[unreadable: τε] φαιν[unreadable: ει]ν [unreadable: επι] [unreadable: της] γης, [unreadable: του] διαχωριζ[unreadable: ει]ν α- [unreadable: polytonic diacritics]\nνα[unreadable: μεσο]ν [unreadable: της] ημερας [unreadable: και] αναμε[unreadable: σο]ν [unreadable: της] νυκ[unreadable: το]ς· [unreadable: και] [unreadable: εστω]σαν [unreadable: ει]ς σημεια, [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] [unreadable: και] εις καιρ[unreadable: ου]ς, [unreadable: και] εις ημε[unreadable: ρα]ς, [unreadable: και] εις ενιαυ[unreadable: του]ς· [unreadable: inline numeral] [unreadable: και] ε[unreadable: στω]σαν εις [unreadable: polytonic diacritics]\nφαυσιν εν τω [unreadable: στε]ρεωματι [unreadable: του] [unreadable: ουρα]νου, ως[unreadable: τε] φαινειν [unreadable: επι] [unreadable: της] γης· [unreadable: και] [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] ε[unreadable: γε]νε[unreadable: το] [unreadable: ου]τως. [unreadable: inline numeral] [unreadable: και] εποιησεν ο θεος [unreadable: τους] δυο φω[unreadable: στ]ηρας [unreadable: τους] με[unreadable: γα]λους, [unreadable: polytonic diacritics]\n[unreadable: το]ν φω[unreadable: στ]ηρα [unreadable: το]ν με[unreadable: γ]αν εις [unreadable: αρ]χας [unreadable: της] ημε[unreadable: ρας]· [unreadable: και] [unreadable: το]ν φω[unreadable: στ]η[unreadable: ρα] [unreadable: το]ν [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] ελα[unreadable: σσ]ω εις [unreadable: αρ]χας [unreadable: της] νυκ[unreadable: το]ς· [unreadable: και] [unreadable: του]ς α[unreadable: στερας]· [unreadable: inline numeral] [unreadable: και] ε[unreadable: θετο] αυ[unreadable: του]ς ο [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] θεος εν τω [unreadable: στε]ρεωμα[unreadable: τι] [unreadable: του] [unreadable: ουρα]νου· ως[unreadable: τε] φαινειν [unreadable: επι] [unreadable: της] γης, [unreadable: inline numeral] [unreadable: και] [unreadable: polytonic diacritics]\nαρχ[unreadable: ει]ν [unreadable: της] ημε[unreadable: ρα]ς [unreadable: και] [unreadable: της] νυκ[unreadable: το]ς· [unreadable: και] διαχωριζ[unreadable: ει]ν α[unreadable: να]με[unreadable: σο]ν [unreadable: του] φω [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] τος [unreadable: και] αναμε[unreadable: σο]ν του σκο[unreadable: του]ς· [unreadable: και] ειδεν ο [unreadable: θεος] ο[unreadable: τι] καλον. [unreadable: inline numeral] [unreadable: και] εγε- [unreadable: polytonic diacritics]\n[unreadable: marginal numeral] νε[unreadable: το] ε[unreadable: σπε]ρα [unreadable: και] ε[unreadable: γε]νε[unreadable: το] π[unreadable: ρω]ι, ημερα [unreadable: τετα]ρτη. [unreadable: inline numeral] [unreadable: και] ειπεν ο θεος, [unreadable: Ε]ξα [unreadable: polytonic diacritics]\nγαγετω [unreadable: τα] υδα[unreadable: τα] ερπετα ψυ[unreadable: χων] ζω[unreadable: σων], [unreadable: και] πετεινα πε[unreadable: τομενα] [unreadable: polytonic diacritics]\n[unreadable: επι] τη[unreadable: ς] γης, [unreadable: κατα] το [unreadable: στε]ρεωμα [unreadable: του] [unreadable: ουρα]νου· [unreadable: και] εγενε[unreadable: το] ουτως. [unreadable: polytonic diacritics]",
      'recovery': 'reviewed-region-text-v1',
      'review_status': 'reviewed-partial',
      'review_recognition': 'approved-transcription-before-ocr',
      'review_source_sha256': source_hash,
      'review_physical_page': '117',
      'review_bounds': str(bounds),
      'review_models': 'gemini-4-argon-high',
      'review_approved_by': 'Codex image review under aarnphm authorization',
      'review_date': '2026-11-04',
      'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md#thinking-with-type-page-117',
      'review_approval': 'Standing marked-partial specimen recovery authorization, 2026-11-03; monitor recovery instruction, 2026-11-04',
    }
  if number == 116 and bounds == [409, 114, 726, 663]:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
      source_hash
      != '728e6d0ca3809d504f12f3302ef5303238a7324cf2824b303d64781d597dba7d'
    ):
      return None
    return {
      'content': '[unreadable: marginal numeral] ב ראשית ברא אלהים את השמים ואת [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] הארץ : [unreadable: inline numeral] והארץ היתה תהו ובהו ו[unreadable: ה/ח]שך [unreadable: vowel and cantillation marks]\nעל־פני תהום ורוח אלהים מרחפת על־ [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] פני המים : [unreadable: inline numeral] ויא[unreadable: מ/כ]ר אלהים יהי־אור [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] ויהי־אור : [unreadable: inline numeral] וירא אלהים את־האור כי־טוב ויב[unreadable: ד/ר]ל [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] אלהים בין האור ובין החשך : [unreadable: inline numeral] ויקרא אלהים לאור [unreadable: vowel and cantillation marks]\nיום ולחשך קרא לילה ויהי־ערב ויהי־בקר יום א[unreadable: ה/ח][unreadable: ד/ר]: [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] [unreadable: inline numeral] ויאמר אלהים יהי רקיע בתוך המים ויהי [unreadable: מ/כ]ב[unreadable: ד/ר]יל [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] בין מים למים : [unreadable: inline numeral] ויעש אלהים את־הרקיע ויב[unreadable: ד/ר]ל בין [unreadable: vowel and cantillation marks]\nהמים אשר מתחת לרקיע ובין המים אשר מעל לרקיע [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] ויהי־כן : [unreadable: inline numeral] ויקרא אלהים לרקיע שמים ויהי־ערב [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] ויהי־בקר יום שני : [unreadable: inline numeral] ויאמר אלהים י[unreadable: ק/ל]וו המים [unreadable: vowel and cantillation marks]\nמתחת השמים אל־[unreadable: מ/כ]קום אח[unreadable: ד/ר] ותראה היבשה ויהי־ [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] כן : [unreadable: inline numeral] ויקרא אלהים ליבשה ארץ ולמקוה ה[unreadable: מ/כ]ים קרא [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] ימים וירא אלהים כי־טוב : [unreadable: inline numeral] ויא[unreadable: מ/כ]ר אלהים ת[unreadable: ד/ר]שא [unreadable: vowel and cantillation marks]\nהארץ [unreadable: ד/ר]שא עשב מזריע זרע עץ פרי עשה פרי ל[unreadable: מ/כ]ינו [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] אשר זרע[unreadable: ו/ר]־בו על־הארץ ויהי־כן : [unreadable: inline numeral] ותוצא הארץ [unreadable: ד/ר]שא [unreadable: vowel and cantillation marks]\nעשב מזריע זרע ל[unreadable: מ/כ]ינהו ועץ עשה־פרי אשר זרעו־בו [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] למינהו וירא אלהים כי־טוב : [unreadable: inline numeral] ויהי־ערב ויהי־[unreadable: ב/כ]קר [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] יום שלישי : [unreadable: inline numeral] ויא[unreadable: מ/כ]ר אלהים יהי [unreadable: מ/כ]ארת ברקיע [unreadable: vowel and cantillation marks]\nהשמים להב[unreadable: ד/ר]יל בין היום ובין הלילה והיו לאתת [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] ולמוע[unreadable: ד/ר]ים ולימים ושנים : [unreadable: inline numeral] והיו למאו[unreadable: רת] ברקיע [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] השמים להאיר על־הארץ ויהי־כן : [unreadable: inline numeral] ויעש אלה[unreadable: י]ם [unreadable: vowel and cantillation marks]\nאת־שני המא[unreadable: רת] הג[unreadable: ד/ר]לים את־המאור הג[unreadable: ד/ר]ל ל[unreadable: ממ/ככ]שלת [unreadable: vowel and cantillation marks]\nהיום ואת־המאור הקטן לממשלת הלילה ואת [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] [unreadable: ה/ח]כוכבים : [unreadable: inline numeral] ויתן אתם אלהים ברקיע השמים להאיר [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] על־הארץ : [unreadable: inline numeral] ולמשל ביום ובלילה ולהב[unreadable: ד/ר]יל בין האור [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] ובין החשך וירא אלהים כי־טוב : [unreadable: inline numeral] ויהי־ערב ויהי־ [unreadable: vowel and cantillation marks]\n[unreadable: marginal numeral] בקר יום רביעי : [unreadable: inline numeral] ויאמר אלהים ישרצו המים שרץ [unreadable: vowel and cantillation marks]\nנפש חיה ועוף יעופף על־הארץ על־פני רקיע השמים : [unreadable: vowel and cantillation marks]',
      'recovery': 'reviewed-region-text-v1',
      'review_status': 'reviewed-partial',
      'review_recognition': 'approved-transcription-before-ocr',
      'review_source_sha256': source_hash,
      'review_physical_page': '116',
      'review_bounds': '[409,114,726,663]',
      'review_models': 'gemini-4-argon-high',
      'review_approved_by': 'Codex image review under aarnphm authorization',
      'review_date': '2026-11-04',
      'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md#thinking-with-type-page-116',
      'review_approval': 'Standing marked-partial specimen recovery authorization, 2026-11-03; monitor recovery instruction, 2026-11-04',
    }
  if number == 116 and bounds == [446, 809, 943, 828]:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
      source_hash
      != '728e6d0ca3809d504f12f3302ef5303238a7324cf2824b303d64781d597dba7d'
    ):
      return None
    return {
      'content': '[unreadable: clipped preceding-line descenders and vowel marks]\n[unreadable: right-clipped word] יו[unreadable: ם/ס] [unreadable: ד/ר][unreadable: ב/כ]יעאי : [unreadable: inline numeral] וא[unreadable: מ/כ/ס][unreadable: ד/ר] [unreadable: ייי] י[unreadable: ד/ר][unreadable: ה/ח]ש[unreadable: ון] [unreadable: מ/כ]יא [unreadable: ד/ר][unreadable: ה/ח]שא נפשא [unreadable: ה/ח]יתא ועופא [unreadable: ד/ר]פ[unreadable: ד/ר][unreadable: ה/ח] על א[unreadable: ד/ר]עא על אפי אוי[unreadable: ד/ר] [unreadable: ד/ר]קיע ש[unreadable: מ/כ]יא : [unreadable: vowel and cantillation marks]',
      'recovery': 'reviewed-region-text-v1',
      'review_status': 'reviewed-partial',
      'review_recognition': 'approved-transcription-before-ocr',
      'review_source_sha256': source_hash,
      'review_physical_page': '116',
      'review_bounds': str(bounds),
      'review_models': 'gemini-4-argon-high',
      'review_approved_by': 'Codex image review under aarnphm authorization',
      'review_date': '2026-11-04',
      'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md#thinking-with-type-page-116',
      'review_approval': 'Standing marked-partial specimen recovery authorization, 2026-11-03; monitor recovery instruction, 2026-11-04',
    }
  if number == 106 and bounds == [151, 432, 414, 489]:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
      source_hash
      != 'cf49bbbb18ec9a1928d32530e9676b16c34d696394579b177ab38dbadc368534'
    ):
      return None
    return {
      'content': '[unreadable: note marker]――[unreadable: katakana]ー[unreadable: katakana][unreadable: katakana][unreadable: katakana]ト・フロ[unreadable: katakana]ト[unreadable: 2 upright chars]\n『[unreadable: kanji][unreadable: kanji][unreadable: kanji]』 S[unreadable: rotated Latin letters] F[unreadable: rotated Latin letters], Th[unreadable: e]\nI[unreadable: rotated Latin letters] [unreadable: of] D[unreadable: rotated Latin letters],\n[unreadable: rotated Latin word] by A. A. B[unreadable: rotated Latin letters],\nG[unreadable: rotated Latin letters] All[unreadable: rotated Latin letters] & U[unreadable: rotated Latin letters],\nL[unreadable: rotated Latin letters], 1[unreadable: 9]1[unreadable: 3], [unreadable: c]h. [unreadable: rotated Roman numeral], p. [unreadable: 3 digits].\n「[unreadable: 4 upright chars]・の[unreadable: 10 upright chars]\n「ー[unreadable: 7 upright chars]、[unreadable: 6 upright chars]\n[unreadable: upright char]の[unreadable: 2 upright chars]」、[unreadable: 10 upright chars]\n[unreadable: 4 upright chars]―[unreadable: 2 upright chars]、の[unreadable: 6 upright chars]\n[unreadable: 6 upright chars]、[unreadable: 5 upright chars]。[unreadable: 3 upright chars]\n[unreadable: 2 upright chars]の[unreadable: 3 upright chars]と[unreadable: 8 upright chars]\n[unreadable: 6 upright chars]。」\n[unreadable: note marker]――[unreadable: 6 katakana chars]・[unreadable: 4 katakana chars]『人[unreadable: kanji]\nの[unreadable: kanji]の[unreadable: 4 kanji]と[unreadable: 4 upright chars]』\nH[unreadable: rotated Latin letters] Sp[unreadable: rotated Latin letters], D[unreadable: ie] S[unreadable: rotated Latin letters]-\n[unreadable: un]d T[unreadable: rotated Latin word] d[unreadable: er]\n[unreadable: rotated Latin word] S[unreadable: rotated Latin letters] [unreadable: rotated Latin word]\nb[unreadable: rotated Latin letters] B[unreadable: rotated Latin word]\n[unreadable: rotated Latin word] V[unreadable: rotated Latin word] [unreadable: 2-letter rotated Latin word] d[unreadable: en]\np[unreadable: rotated Latin word] Al[unreadable: rotated Latin letters], [unreadable: 2]n[unreadable: d]\n[unreadable: rotated Latin abbreviation]., F[unreadable: rotated Latin letters] F[unreadable: rotated Latin letters], T[unreadable: rotated Latin letters],\n1[unreadable: 3 digits], [unreadable: c]h. [unreadable: rotated Roman numeral], p. [unreadable: 3 digits]. 「[unreadable: 5 upright chars]\n[unreadable: 2 upright chars]に[unreadable: 4 upright chars]と[unreadable: upright char]は[unreadable: 5 upright chars]\nもの[unreadable: て/で]は、[unreadable: 3 upright chars]なもの[unreadable: て/で]も[unreadable: 2 upright chars]\n」、[unreadable: 3 upright chars]は[unreadable: 9 upright chars]。\n――[unreadable: そ]れに[unreadable: 2 upright chars]て一[unreadable: 6 upright chars]\n[unreadable: 3 upright chars]。」\n[unreadable: note marker]――[unreadable: 4 katakana chars]ト・[unreadable: 5 katakana chars]ト\n『[unreadable: kanji]について』 R[unreadable: rotated Latin letters] D[unreadable: rotated Latin letters],\nE[unreadable: rotated Latin letters] [unreadable: on] [unreadable: th]e Th[unreadable: rotated Latin letters] [unreadable: o]f\nN[unreadable: rotated Latin letters], [unreadable: rotated Latin word] by\nW[unreadable: rotated Latin letters] W[unreadable: rotated Latin letters] B[unreadable: rotated Latin letters],',
      'recovery': 'reviewed-region-text-v1',
      'review_status': 'reviewed-partial',
      'review_recognition': 'approved-transcription-before-ocr',
      'review_source_sha256': source_hash,
      'review_physical_page': '106',
      'review_bounds': str(bounds),
      'review_models': 'gemini-4-argon-high',
      'review_approved_by': 'Codex image review under aarnphm authorization',
      'review_date': '2026-11-04',
      'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md#grid-systems-page-106',
      'review_approval': 'Standing marked-partial specimen recovery authorization, 2026-11-03; monitor recovery instruction, 2026-11-04',
    }
  if number == 114 and bounds == [87, 399, 169, 430]:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
      source_hash
      != 'cf49bbbb18ec9a1928d32530e9676b16c34d696394579b177ab38dbadc368534'
    ):
      return None
    return {
      'content': '[unreadable: miniature brochure specimen; literal glyph recognition unresolved at source resolution]',
      'recovery': 'reviewed-region-text-v1',
      'review_status': 'reviewed-partial',
      'review_recognition': 'approved-transcription-before-ocr',
      'review_source_sha256': source_hash,
      'review_physical_page': '114',
      'review_bounds': str(bounds),
      'review_models': 'gemini-4-argon-high',
      'review_approved_by': 'Codex image review under aarnphm authorization',
      'review_date': '2026-11-04',
      'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md#grid-systems-page-114',
      'review_approval': 'Standing marked-partial specimen recovery authorization, 2026-11-03; monitor recovery instruction, 2026-11-04',
    }
  if number == 109 and bounds == [229, 378, 301, 408]:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
      source_hash
      != 'cf49bbbb18ec9a1928d32530e9676b16c34d696394579b177ab38dbadc368534'
    ):
      return None
    return {
      'content': '[unreadable: miniature brochure specimen; literal glyph recognition unresolved at source resolution]',
      'recovery': 'reviewed-region-text-v1',
      'review_status': 'reviewed-partial',
      'review_recognition': 'approved-transcription-before-ocr',
      'review_source_sha256': source_hash,
      'review_physical_page': '109',
      'review_bounds': str(bounds),
      'review_models': 'gemini-4-argon-high',
      'review_approved_by': 'Codex image review under aarnphm authorization',
      'review_date': '2026-11-04',
      'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md#grid-systems-page-109',
      'review_approval': 'Standing marked-partial specimen recovery authorization, 2026-11-03; monitor recovery instruction, 2026-11-04',
    }
  if number == 107 and bounds == [224, 286, 329, 395]:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
      source_hash
      != 'cf49bbbb18ec9a1928d32530e9676b16c34d696394579b177ab38dbadc368534'
    ):
      return None
    return {
      'content': '[unreadable: kanji][unreadable: kanji]\u3000アナーキー[unreadable: た/だ]、と[unreadable: kanji][unreadable: kanji][unreadable: を][unreadable: マ]クロにと\nらえた[unreadable: kanji][unreadable: kanji][unreadable: kanji][unreadable: か/が]、[unreadable: kanji][unreadable: 代]の[unreadable: kanji][unreadable: kanji][unreadable: kanji]というの[unreadable: て/で]\nし[unreadable: ょ]うね。た[unreadable: た/だ][unreadable: kanji][unreadable: kanji]と[unreadable: kanji][unreadable: kanji]の[unreadable: kanji]いは、[unreadable: kanji][unreadable: kanji]\nは[unreadable: kanji][unreadable: て/で][unreadable: kanji][unreadable: ま]り、[unreadable: kanji][unreadable: て/で][unreadable: kanji][unreadable: わ]る。[unreadable: kanji][unreadable: kanji]も[unreadable: kanji][unreadable: を][unreadable: kanji]\nしてはいるけれ[unreadable: と/ど]も、[unreadable: 常]に[unreadable: kanji][unreadable: kanji]には[unreadable: パ][unreadable: ブ]リ\n[unreadable: ッ]クなものに[unreadable: kanji][unreadable: kanji]するために[unreadable: kanji][unreadable: か/が]フィルタ\nーにな[unreadable: っ]ている。[unreadable: そ]この[unreadable: kanji]いは[unreadable: あ/お]る[unreadable: て/で]し[unreadable: ょ]\nうね、き[unreadable: っ]と。',
      'recovery': 'reviewed-region-text-v1',
      'review_status': 'reviewed-partial',
      'review_recognition': 'approved-transcription-before-ocr',
      'review_source_sha256': source_hash,
      'review_physical_page': '107',
      'review_bounds': str(bounds),
      'review_models': 'gemini-4-argon-high',
      'review_approved_by': 'Codex image review under aarnphm authorization',
      'review_date': '2026-11-04',
      'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md#grid-systems-page-107',
      'review_approval': 'Standing marked-partial specimen recovery authorization, 2026-11-03; monitor recovery instruction, 2026-11-04',
    }
  if number == 107 and bounds == [382, 287, 489, 395]:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
      source_hash
      != 'cf49bbbb18ec9a1928d32530e9676b16c34d696394579b177ab38dbadc368534'
    ):
      return None
    return {
      'content': '[unreadable: kanji]田\u3000[unreadable: そ]れはこう[unreadable: し/じ][unreadable: ゃ]ないかという[unreadable: kanji][unreadable: か/が]す\nる。[unreadable: kanji][unreadable: kanji]な[unreadable: と/ど]は日本[unreadable: て/で]は[unreadable: 古]くは[unreadable: kanji][unreadable: kanji]、[unreadable: あ/お]る\nいは[unreadable: あ/お][unreadable: kanji][unreadable: kanji]から[unreadable: kanji][unreadable: ま][unreadable: っ]た。[unreadable: そ]れ[unreadable: か/が][unreadable: kanji][unreadable: kanji][unreadable: kanji][unreadable: kanji]\n[unreadable: か/が]いろいろ[unreadable: 生][unreadable: ま]れると、[unreadable: kanji][unreadable: kanji]にな[unreadable: っ]たり[unreadable: kanji]\n[unreadable: kanji]にな[unreadable: っ]たり、大正の[unreadable: kanji][unreadable: kanji]の[unreadable: 弁][unreadable: 士/土]に[unreadable: ま][unreadable: て/で][unreadable: kanji]\n[unreadable: kanji]する。[unreadable: 文][unreadable: kanji]も分[unreadable: 化]する[unreadable: わ]け[unreadable: て/で]し[unreadable: ょ]う。[unreadable: エ]\n[unreadable: ッ]セイ[unreadable: 的]なもの、小[unreadable: kanji][unreadable: 的]なもの、[unreadable: そ]の小[unreadable: kanji]\nもいろいろ分かれていく。',
      'recovery': 'reviewed-region-text-v1',
      'review_status': 'reviewed-partial',
      'review_recognition': 'approved-transcription-before-ocr',
      'review_source_sha256': source_hash,
      'review_physical_page': '107',
      'review_bounds': str(bounds),
      'review_models': 'gemini-4-argon-high',
      'review_approved_by': 'Codex image review under aarnphm authorization',
      'review_date': '2026-11-04',
      'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md#grid-systems-page-107',
      'review_approval': 'Standing marked-partial specimen recovery authorization, 2026-11-03; monitor recovery instruction, 2026-11-04',
    }
  if number == 106 and bounds == [341, 281, 417, 427]:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
      source_hash
      != 'cf49bbbb18ec9a1928d32530e9676b16c34d696394579b177ab38dbadc368534'
    ):
      return None
    return {
      'content': 'と[unreadable: kanji][unreadable: kanji][unreadable: kanji][unreadable: kanji]に[unreadable: あ/お]る[unreadable: kanji][unreadable: kanji]の[unreadable: 区][unreadable: kanji][unreadable: か/が][unreadable: て/で][unreadable: き/さ]ない、ということに[unreadable: kanji][unreadable: kanji][unreadable: か/が]\n一[unreadable: kanji]しているらしい[unreadable: 2 emphasis dots (wakiten)]。\n「[unreadable: kanji]」と「[unreadable: kanji]」と[unreadable: を][unreadable: kanji][unreadable: べ]て、い[unreadable: っ]し[unreadable: ょ]に[unreadable: kanji][unreadable: し/じ]ることは、[unreadable: kanji]た[unreadable: 目/日]には\n[unreadable: kanji][unreadable: kanji]な[unreadable: kanji][unreadable: し/じ][unreadable: を][unreadable: kanji][unreadable: え]るし、この[unreadable: kanji]の[unreadable: kanji][unreadable: kanji]に[unreadable: 引]いた[unreadable: kanji][unreadable: kanji]の[unreadable: 主/王][unreadable: て/で][unreadable: あ/お]る\n[unreadable: 亡][unreadable: kanji]の[unreadable: kanji][unreadable: り][unreadable: を][unreadable: kanji][unreadable: め/ぬ]る[unreadable: た/だ][unreadable: け]のた[unreadable: め/ぬ]にも、なんらかの[unreadable: kanji][unreadable: kanji][unreadable: か/が][unreadable: kanji][unreadable: kanji]か\nもしれない。この[unreadable: kanji][unreadable: kanji]は[unreadable: 図][unreadable: kanji][unreadable: enclosed numeral]のうちに[unreadable: あ/お]る[unreadable: kanji][unreadable: kanji][unreadable: 示][unreadable: さ/き]れている[unreadable: bottom-clipped 。]',
      'recovery': 'reviewed-region-text-v1',
      'review_status': 'reviewed-partial',
      'review_recognition': 'approved-transcription-before-ocr',
      'review_source_sha256': source_hash,
      'review_physical_page': '106',
      'review_bounds': str(bounds),
      'review_models': 'gemini-4-argon-high',
      'review_approved_by': 'Codex image review under aarnphm authorization',
      'review_date': '2026-11-04',
      'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md#grid-systems-page-106',
      'review_approval': 'Standing marked-partial specimen recovery authorization, 2026-11-03; monitor recovery instruction, 2026-11-04',
    }
  if number == 105 and bounds in ([149, 385, 233, 440], [328, 386, 412, 435]):
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
      source_hash
      != 'cf49bbbb18ec9a1928d32530e9676b16c34d696394579b177ab38dbadc368534'
    ):
      return None
    return {
      'content': '[unreadable: miniature brochure specimen; literal glyph recognition unresolved at source resolution]',
      'recovery': 'reviewed-region-text-v1',
      'review_status': 'reviewed-partial',
      'review_recognition': 'approved-transcription-before-ocr',
      'review_source_sha256': source_hash,
      'review_physical_page': '105',
      'review_bounds': str(bounds),
      'review_models': 'gemini-4-argon-high',
      'review_approved_by': 'Codex image review under aarnphm authorization',
      'review_date': '2026-11-04',
      'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md#grid-systems-page-105',
      'review_approval': 'Standing marked-partial specimen recovery authorization, 2026-11-03; monitor recovery instruction, 2026-11-04',
    }
  if number == 97 and bounds == [267, 340, 376, 420]:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
      source_hash
      != 'cf49bbbb18ec9a1928d32530e9676b16c34d696394579b177ab38dbadc368534'
    ):
      return None
    return {
      'content': '[unreadable: miniature German annual-report specimen, 16 lines; literal glyph recognition unresolved at source resolution]',
      'recovery': 'reviewed-region-text-v1',
      'review_status': 'reviewed-partial',
      'review_recognition': 'approved-transcription-before-ocr',
      'review_source_sha256': source_hash,
      'review_physical_page': '97',
      'review_bounds': '[267,340,376,420]',
      'review_models': 'gemini-4-argon-high',
      'review_approved_by': 'Codex image review under aarnphm authorization',
      'review_date': '2026-11-04',
      'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md#grid-systems-page-97',
      'review_approval': 'Standing marked-partial specimen recovery authorization, 2026-11-03; monitor recovery instruction, 2026-11-04',
    }
  if number == 175 and bounds == [719, 584, 904, 606]:
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
      source_hash
      != 'dfc364b1cdc0e71f18a2f6a5bf4b4b5660fb4de2f93a155e4cae949f3e9ae991'
    ):
      return None
    return {
      'content': (
        '[unreadable: clipped preceding-line fragment]\n'
        'Pretty soon, it was a [unreadable: miniature specimen words] place, '
        '[unreadable: miniature specimen words]\n'
        '[unreadable: miniature specimen words] and '
        '[unreadable: miniature specimen words], and people began to '
        '[unreadable: miniature specimen words] around\n'
        '[unreadable: miniature specimen words] to look at what was going\n'
        'on in the city.'
      ),
      'recovery': 'reviewed-region-text-v1',
      'review_status': 'reviewed-partial',
      'review_recognition': 'approved-transcription-before-ocr',
      'review_source_sha256': source_hash,
      'review_physical_page': '175',
      'review_bounds': '[719,584,904,606]',
      'review_models': 'gemini-4-argon-high',
      'review_approved_by': 'Codex image review under aarnphm authorization',
      'review_date': '2026-11-04',
      'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md#ia2001-page-175',
      'review_approval': 'Standing marked-partial specimen recovery authorization, 2026-11-03',
    }
  expected = '0caa6b96a86bb319a6a577500705cb8072da1b96341d46f35624ea48d8a80a49'
  if number == 96 and bounds == [155, 514, 222, 534]:
    if content not in (None, ''):
      return None
    text = 'Forms'
    models = 'gemini-4-argon-high,claude-opus-5-5-high'
  elif number == 12 and bounds == [256, 825, 766, 912]:
    text = (
      'begin  new max, min;\n'
      "  bop max ←'new a,b. if a>b then a else b';\n"
      "  bop min ←'new a,b. if a<b then a else b';"
    )
    models = 'gemini-4-argon-high'
  elif number == 29 and bounds == [80, 227, 239, 390]:
    if content not in (None, ''):
      return None
    expected = (
      '728e6d0ca3809d504f12f3302ef5303238a7324cf2824b303d64781d597dba7d'
    )
    text = 'a'
    models = 'gemini-4-argon-high'
  elif number == 29 and bounds == [280, 305, 866, 397]:
    if content is None:
      return None
    expected = (
      '728e6d0ca3809d504f12f3302ef5303238a7324cf2824b303d64781d597dba7d'
    )
    text = 'abcdefghi jklmnopqr\ntuvwxyz ÿ'
    models = 'gemini-4-argon-high'
  elif number == 97 and bounds in (
    [0, 269, 82, 414],
    [0, 397, 101, 469],
    [0, 435, 103, 490],
    [0, 501, 190, 615],
    [4, 548, 218, 672],
    [13, 596, 243, 728],
    [163, 152, 543, 401],
    [155, 440, 365, 640],
    [347, 432, 391, 453],
    [316, 441, 458, 605],
    [410, 399, 676, 548],
  ):
    if content not in (None, ''):
      return None
    expected = (
      '728e6d0ca3809d504f12f3302ef5303238a7324cf2824b303d64781d597dba7d'
    )
    text = (
      '[unreadable: photographed German specimen; recognition unresolved '
      'in this region]'
    )
    models = 'gemini-4-argon-high'
  elif number == 57 and bounds in ([114, 301, 286, 641], [292, 304, 464, 642]):
    if content not in (None, ''):
      return None
    expected = (
      'cf49bbbb18ec9a1928d32530e9676b16c34d696394579b177ab38dbadc368534'
    )
    text = '[unreadable: dense German typography specimen; complete literal transcription remains unresolved]'
    models = 'gemini-4-argon-high'
  elif bounds in {
    58: [[190, 299, 538, 638]],
    59: [[76, 275, 291, 691], [288, 276, 505, 687]],
    60: [[165, 294, 516, 632], [585, 295, 938, 634]],
    61: [[91, 301, 443, 642]],
    63: [
      [196, 386, 275, 455],
      [196, 462, 276, 516],
      [195, 540, 279, 611],
      [106, 615, 274, 650],
      [375, 387, 457, 453],
      [376, 462, 459, 512],
      [376, 539, 460, 607],
      [287, 614, 457, 649],
      [526, 610, 606, 643],
      [614, 610, 696, 643],
      [707, 457, 789, 525],
      [798, 609, 879, 644],
      [707, 609, 790, 643],
    ],
  }.get(number, []):
    if content not in (None, ''):
      return None
    expected = (
      'cf49bbbb18ec9a1928d32530e9676b16c34d696394579b177ab38dbadc368534'
    )
    text = '[unreadable: miniature typography specimen; complete literal transcription remains unresolved]'
    models = 'gemini-4-argon-high'
  elif number == 150 and bounds == [84, 465, 351, 853]:
    if content not in (None, ''):
      return None
    expected = (
      'dfc364b1cdc0e71f18a2f6a5bf4b4b5660fb4de2f93a155e4cae949f3e9ae991'
    )
    text = '[unreadable: miniature restaurant-guide specimen; complete literal transcription remains unresolved]'
    models = 'gemini-4-argon-high'
  elif number == 114 and bounds == [444, 203, 703, 817]:
    if content not in (None, ''):
      return None
    expected = (
      '728e6d0ca3809d504f12f3302ef5303238a7324cf2824b303d64781d597dba7d'
    )
    text = '[unreadable: blackletter Latin commentary; complete literal transcription remains unresolved]'
    models = 'gemini-4-argon-high'
  elif number == 804 and bounds == [223, 818, 269, 827]:
    expected = (
      '13051c1e0401281930ccc533cbcdabcfec77a3eb69fd201e3bbd4796b4a91d07'
    )
    text = '. . . .'
    models = 'gemini-4-argon-high'
  else:
    return None
  source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
  if source_hash != expected:
    return None
  reviewed = {
    'content': text,
    'recovery': 'reviewed-region-text-v1',
    'review_source_sha256': source_hash,
    'review_reference': 'docs/verification/ocr-aside-recovery/source-review.md',
    'review_models': models,
    'review_approved_by': 'aarnphm',
    'review_date': '2026-11-02',
  }
  if number in (12, 57, 58, 59, 60, 61, 63, 114, 150, 804) or (
    number == 97 and bounds == [0, 269, 82, 414]
  ):
    reviewed['review_recognition'] = 'approved-transcription-before-ocr'
  if number in (29, 804):
    reviewed['review_approved_by'] = (
      'Codex image review under aarnphm authorization'
    )
    reviewed['review_date'] = '2026-11-03'
  if number == 97:
    reviewed['review_status'] = 'reviewed-partial'
    reviewed['review_date'] = '2026-11-03'
    reviewed['review_reference'] = (
      'docs/verification/ocr-aside-recovery/source-review.md#thinking-with-type-page-97-authorized-partial-ocr'
    )
  if number in (57, 58, 59, 60, 61, 63, 114, 150):
    reviewed['review_status'] = 'reviewed-partial'
    reviewed['review_date'] = '2026-11-03'
    reviewed['review_reference'] = (
      'docs/verification/ocr-aside-recovery/source-review.md#authorized-partial-ocr-for-remaining-blocked-specimens'
    )
  return reviewed


def reviewed_scan_artifact(
  source: Path, number: int, bounds: list[float]
) -> bool:
  # Exact scan-edge boxes verified against full pages and enlarged crops.
  # Visual evidence and replay: docs/verification/ocr-aside-recovery/source-review.md.
  if number == 18 and bounds == [9, 269, 27, 468]:
    expected = (
      '8a4f49282b4843078f7935c77b232a7ad06f258f4d2479691b74c38ed68e698a'
    )
  elif number == 36 and bounds == [34, 541, 51, 652]:
    expected = (
      '52d6338905af0ab30081fc02e8b998f50e813495e54e2d0e28a7283c97553fee'
    )
  else:
    return False
  return hashlib.sha256(source.read_bytes()).hexdigest() == expected


def vector_aside(source: Path, number: int, bounds: list[float]) -> bool:
  with pymupdf.open(source) as pdf:
    page = pdf[number - 1]
    if not page.get_text().strip():
      return False
    area = (
      pymupdf.Rect(
        bounds[0] * page.rect.width / 1000,
        bounds[1] * page.rect.height / 1000,
        bounds[2] * page.rect.width / 1000,
        bounds[3] * page.rect.height / 1000,
      )
      * page.derotation_matrix
    )
    # Detector bounds are quantized to a 0-1000 grid. Allow one grid unit
    # at each edge, including in the native-text exclusion check.
    tolerance = (
      pymupdf.Rect(
        max(0, bounds[0] - 1) * page.rect.width / 1000,
        max(0, bounds[1] - 1) * page.rect.height / 1000,
        min(1000, bounds[2] + 1) * page.rect.width / 1000,
        min(1000, bounds[3] + 1) * page.rect.height / 1000,
      )
      * page.derotation_matrix
    )
    if area.is_empty or any(
      pymupdf.Rect(word[:4]).intersects(tolerance)
      for word in page.get_text('words')
    ):
      return False
    for drawing in page.get_drawings():
      box = drawing['rect']
      if tolerance.contains(box) and box.get_area() >= area.get_area() * 0.5:
        return True
    return False


def native_region_text(
  source: Path, number: int, bounds: list[float]
) -> str | None:
  """Recover complete visible native lines, retaining exponent/base markers."""

  with pymupdf.open(source) as pdf:
    page = pdf[number - 1]
    area = (
      pymupdf.Rect(
        bounds[0] * page.rect.width / 1000,
        bounds[1] * page.rect.height / 1000,
        bounds[2] * page.rect.width / 1000,
        bounds[3] * page.rect.height / 1000,
      )
      * page.derotation_matrix
    )
    margin = 8 * max(page.rect.width, page.rect.height) / 1000
    tolerance = pymupdf.Rect(
      area.x0 - margin, area.y0 - margin, area.x1 + margin, area.y1 + margin
    )
    for span in page.get_texttrace():
      if (span['type'] != 0 or span['opacity'] == 0) and pymupdf.Rect(
        span['bbox']
      ).intersects(area):
        return None
    for image in page.get_image_info():
      overlap = pymupdf.Rect(image['bbox']) & area
      if overlap.get_area() > area.get_area() * 0.05:
        return None
    lines: list[str] = []
    for block in page.get_text('dict')['blocks']:
      for line in block.get('lines', []):
        box = pymupdf.Rect(line['bbox'])
        center = (box.tl + box.br) / 2
        if not area.contains(center):
          continue
        if not tolerance.contains(box) or line['dir'] != (1.0, 0.0):
          return None
        spans = line['spans']
        baseline = max(spans, key=lambda span: span['size'])
        fragments: list[str] = []
        for span in spans:
          text = span['text']
          if '\ufffd' in text or any(ord(char) < 32 for char in text):
            return None
          if span['flags'] & 1:
            text = '^{' + text + '}'
          elif (
            span['size'] < baseline['size'] * 0.85
            and span['origin'][1]
            > baseline['origin'][1] + baseline['size'] * 0.1
          ):
            text = '_{' + text + '}'
          fragments.append(text)
        lines.append(''.join(fragments).strip())
    text = '\n'.join(lines).strip()
    return text or None


def native_binary_text(
  source: Path, number: int, bounds: list[float]
) -> str | None:
  text = native_region_text(source, number, bounds)
  if text is None or not re.search(r'(?:[01]{8}[ \t]+){7}[01]{8}', text):
    return None
  return text
