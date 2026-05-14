import os
import sys
import time
import re
import threading
from dotenv import load_dotenv
from google import genai
import anthropic

# Windows console: force UTF-8 so emojis + Arabic don't crash on cp1252
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()

# 1. إعداد العملاء
GEMINI_API_KEY = "AIzaSyAZsXXoJz5sYaTZh_d9mF7667431K_u8XI"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# 2. الثوابت الجوهرية (المصدر الحقيقي لمنع الهلوسة)
FACTS_CONTEXT = """
حقائق كأس العالم 2026 المعتمدة:
- إجمالي المباريات: 104 مباريات (72 في دور المجموعات).
- النظام: 48 منتخباً في 12 مجموعة.
- الافتتاح: 11 يونيو 2026 في ملعب أزتيكا بالمكسيك.
- النهائي: 19 يوليو 2026 في ملعب ميتلايف بنيويورك/نيوجيرسي.
- المنتخبات العربية (8): السعودية، قطر، المغرب، تونس، مصر، الجزائر، الأردن، العراق.
- مجموعة السعودية (H): إسبانيا، أوروغواي، الرأس الأخضر.
- مجموعة المغرب (C): البرازيل، هايتي، اسكتلندا.
- قوانين جديدة: قانون 'الكابتن فقط'، قاعدة الدقيقة للمصاب، 5 ثوانٍ لرميات التماس.
"""

# Gemini → أسئلة حقائق مباشرة (مواعيد، ملاعب، مجموعات، قوانين)
GEMINI_PROMPT = f"""أنت مهندس بيانات لنموذج "علام". استخدم الحقائق التالية فقط:
{FACTS_CONTEXT}

المهمة: توليد 50 زوج سؤال-جواب بصيغة JSONL.
زاوية الأسئلة: حقائق مباشرة فقط (مواعيد، ملاعب، أرقام، مجموعات، قوانين، أسماء منتخبات).
أمثلة على النوع المطلوب: "متى تبدأ البطولة؟"، "كم عدد المباريات؟"، "وين يلعب المنتخب السعودي؟".
ممنوع: أسئلة الرأي، التوقعات، المقارنات، السيناريوهات.

التوزيع: 50% فصحى، 20% عامية سعودية، 30% إنجليزية.
الأجوبة: محادثاتية، طبيعية، وقصيرة جداً.
الصيغة المطلوبة لكل سطر: {{"instruction": "السؤال", "input": "", "output": "الجواب"}}
ممنوع إضافة أي نص ترحيبي أو علامات Markdown."""

# Claude → أسئلة تحليلية وسيناريوهات ومقارنات (مكملة لـ Gemini وغير متشابهة)
CLAUDE_PROMPT = f"""أنت مهندس بيانات لنموذج "علام". استخدم الحقائق التالية فقط ولا تختلق أي معلومة خارجها:
{FACTS_CONTEXT}

المهمة: توليد 50 زوج سؤال-جواب بصيغة JSONL.
زاوية الأسئلة (مهم جداً - مختلفة عن أسئلة الحقائق المباشرة):
- سيناريوهات وتجارب جماهيرية: "ابي أحضر مباراة السعودية، وش أسوي؟"، "كيف أوصل لملعب أزتيكا؟"
- مقارنات بين المجموعات والمنتخبات: "أيهما أصعب مجموعة السعودية أو المغرب؟"
- شرح وتفسير القوانين الجديدة: "وش معنى قانون الكابتن فقط؟ ومتى يطبق؟"
- توقعات مبنية على الحقائق فقط: "مين أقوى منتخب في مجموعة المغرب حسب الترتيب العالمي؟"
- أسئلة محادثاتية مفتوحة عن البطولة والتنظيم
- تحديات تواجه المنتخبات العربية الـ 8

ممنوع منعاً باتاً تكرار صيغة الأسئلة المباشرة مثل "متى تبدأ؟" أو "كم عدد المباريات؟" - هذي الأسئلة موجودة في dataset ثاني.

التوزيع: 50% فصحى، 20% عامية سعودية/خليجية، 30% إنجليزية.
الأجوبة: محادثاتية، طبيعية، متوسطة الطول (سطر إلى سطرين)، مبنية على الحقائق المعطاة فقط.
الصيغة المطلوبة لكل سطر: {{"instruction": "السؤال", "input": "", "output": "الجواب"}}
ممنوع إضافة أي نص ترحيبي أو علامات Markdown."""


def clean_response(text):
    """تنظيف الرد من علامات Markdown ومخلفات النص"""
    text = re.sub(r'```jsonl?|```', '', text)
    return text.strip()


GEMINI_MODEL = "gemini-2.5-flash"


def log(tag, msg):
    print(f"[{tag}] {msg}", flush=True)


def gemini_worker(max_batches=100, max_consecutive_quota=5):
    output_path = "data/synthetic/qa_gemini.jsonl"
    batch = 1
    consecutive_quota = 0
    log("GEMINI", f"🚀 بدء التوليد - موديل {GEMINI_MODEL} - الهدف {max_batches} دفعة")

    with open(output_path, "a", encoding="utf-8") as f:
        while batch <= max_batches:
            try:
                response = gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=GEMINI_PROMPT,
                )
                if response.text:
                    f.write(clean_response(response.text) + "\n")
                    f.flush()
                    log("GEMINI", f"✅ {batch}/{max_batches}")
                    batch += 1
                    consecutive_quota = 0  # نجحت → صفّر العدّاد
                else:
                    log("GEMINI", "⚠️ رد فارغ من الـ API - إعادة محاولة")
                time.sleep(8)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    consecutive_quota += 1
                    if consecutive_quota >= max_consecutive_quota:
                        log(
                            "GEMINI",
                            f"❌ توقّف نهائي - {max_consecutive_quota} quota errors متتالية. "
                            f"غالباً ضرب الحد اليومي. Claude سيكمل وحده.",
                        )
                        break
                    log(
                        "GEMINI",
                        f"⏳ كوتا ({consecutive_quota}/{max_consecutive_quota}).. انتظار 60 ثانية",
                    )
                    time.sleep(60)
                elif "404" in error_str or "NOT_FOUND" in error_str:
                    log("GEMINI", f"❌ خطأ 404 - الموديل {GEMINI_MODEL} غير متاح: {e}")
                    break
                elif "401" in error_str or "403" in error_str or "API_KEY" in error_str:
                    log("GEMINI", f"❌ مشكلة في المفتاح: {e}")
                    break
                else:
                    log("GEMINI", f"⚠️ {type(e).__name__}: {e}")
                    time.sleep(10)
    log("GEMINI", "🏁 انتهى worker")


def claude_worker(max_batches=100):
    output_path = "data/synthetic/qa_claude.jsonl"
    batch = 1
    log("CLAUDE", f"🚀 بدء التوليد - الهدف {max_batches} دفعة")

    with open(output_path, "a", encoding="utf-8") as f:
        while batch <= max_batches:
            try:
                response = claude_client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=4096,
                    messages=[{"role": "user", "content": CLAUDE_PROMPT}],
                )
                text = response.content[0].text
                if text:
                    f.write(clean_response(text) + "\n")
                    f.flush()
                    log("CLAUDE", f"✅ {batch}/{max_batches}")
                    batch += 1
                time.sleep(2)
            except anthropic.RateLimitError:
                log("CLAUDE", "⏳ rate limit.. انتظار 60 ثانية")
                time.sleep(60)
            except Exception as e:
                log("CLAUDE", f"⚠️ {type(e).__name__}: {e}")
                time.sleep(10)


def generate_data():
    os.makedirs("data/synthetic", exist_ok=True)
    print("🚀 بدء مصنع البيانات لمونديال 2026 (Gemini + Claude بالتوازي)")

    threads = [
        threading.Thread(target=gemini_worker, name="gemini", daemon=False),
        threading.Thread(target=claude_worker, name="claude", daemon=False),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("🎉 انتهى التوليد من الموديلين")


if __name__ == "__main__":
    generate_data()
