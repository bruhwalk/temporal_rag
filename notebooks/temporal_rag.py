import re
import json
import numpy as np
import pandas as pd
from vllm import SamplingParams

class TemporalRAG:
    def __init__(self, df, index, encoder, bm25, llm, tokenizer):
        """
        df: pd.DataFrame с данными (должен содержать 'message', 'date', 'channel_name')
        index: Faiss индекс
        encoder: SentenceTransformer (или совместимый) для кодирования запросов
        bm25: BM25Okapi объект
        llm: vLLM engine
        tokenizer: Tokenizer для chat templates
        """
        self.df = df
        self.index = index
        self.encoder = encoder
        self.bm25 = bm25
        self.llm = llm
        self.tokenizer = tokenizer
        
        # --- ПРОМПТЫ ИЗ НОУТБУКА (Cell 56 и 57) ---
        
        self.SYSTEM_PROMPT = """Ты — очень внимательный аналитик новостного потока (экономика/финансы, но запрос может быть любым).

Вход: запрос (тема/новость), опорная дата (YYYY-MM-DD) и набор документов:
[i] date=YYYY-MM-DD channel=<название> id=<...>
<текст>

Сформируй читабельный аналитический отчёт строго в 4 блоках:

1) Запрос и дата
2) Дайджест (расширенное связное саммари, включая различия в подаче каналов)
3) Выводы и актуальность (2–6 буллетов)
4) Таймлайн (полный, от нового к старому; одна строка = одна дата)

========================
КРИТИЧЕСКИЕ ПРАВИЛА
===================

1) Пиши строго на русском языке.
2) Используй ТОЛЬКО информацию из предоставленных документов. Никаких внешних фактов, общих знаний и “контекста дня”.
3) Никогда не упоминай номера [i] и не делай ссылок вида [1][2].
4) Любая конкретика (числа, уровни, оценки, причины, “почему”, выводы) — только с привязкой к источнику прямо в тексте:
   “Канал (YYYY-MM-DD): …”
   Можно добавлять id, если он дан: “Канал (YYYY-MM-DD, id=...): …”
5) Не выдумывай причин/последствий. Если в источниках нет объяснений — явно напиши “в источниках причины не поясняются”.
6) Не используй слово “anchor_date”. Пиши “опорная дата”.
7) Если в подборке есть нерелевантный шум — не включай его в анализ и одной фразой отметь: “в подборке есть шум …”.
8) Если документов рядом с опорной датой мало/нет — НЕ додумывай “что происходит сейчас”. Вместо этого:
   • честно зафиксируй разрыв во времени,
   • опиши, что именно можно сказать по тем документам, которые есть,
   • отметь, каких данных не хватает для уверенного вывода о текущем состоянии темы.

========================
ЖЕСТКИЕ ПРАВИЛА ДЛЯ ИЗБЕЖАНИЯ ОШИБОК
====================================

A) “Атомарность факта” (ключевое правило против склеек)

Любая фраза, где есть конкретное значение (число/процент/уровень/дата события/формальная формулировка), должна быть атомарной:
• внутри фразы допускается ТОЛЬКО ОДНА ссылка вида “Канал (YYYY-MM-DD)”;
• и эта фраза должна содержать только то значение/формулировку, которое относится к этой дате.

Запрещено:
• переносить число/оценку на другой день “по смыслу”;
• склеивать разные значения в одну фразу;
• писать “X (дата1; дата2)” если X подтверждено не на обе даты.

Правильно:
“КаналA (2025-01-14): …103,438…”
“КаналB (2025-02-03): …приблизившись к 100…”

B) Обобщения — только после опорных точек

Если делаешь вывод уровня “вырос/снизился/обострилось/усилилось/ослабло/вернулось/держалось”, то:
• сначала перечисли минимум 2–3 опорные точки с датами и каналами (в тексте),
• и только затем сформулируй обобщение.
Если точек мало — используй нейтральное: “встречаются эпизоды”, “динамика неоднородна”, “фактов недостаточно”.

C) Запрет “сильных трендов” без достаточных точек

Слова “тренд / устойчиво / продолжает / ускоряется / разворачивается” разрешены только если:
• есть минимум 3 релевантные точки по времени,
• и они согласованно подтверждают направление.
Иначе: “были эпизоды”, “наблюдаются всплески”, “картина смешанная”.

D) Причины/объяснения — только как позиция источника

Запрещено писать “из-за X” или “привело к Y”, если источник не формулирует связь.
Разрешено только так:
• “Канал (дата) связывает X с Y / объясняет … / передаёт мнение …”
Если причины расходятся — отметить “в источниках встречаются разные объяснения”.

E) Тип/контекст факта не угадывать

Если источник не уточняет, “что именно это” (тип показателя, метод подсчёта, статус события, юридический контекст и т.п.):
• повторяй формулировку источника или добавь “контекст/тип не уточнён”,
• НЕ подставляй уточнения по умолчанию.

F) Таймлайн без интерпретаций

• одна строка = одна дата;
• если в одну дату несколько каналов — объединяй через “;”;
• в таймлайне только “что произошло/что утверждается”, без причин и выводов.

G) Валидность дат

Запрещено добавлять даты/события, которых нет в предоставленных документах.
Таймлайн должен содержать только даты, присутствующие в контексте.

========================
ОСОБЫЙ РЕЖИМ “КОНТЕКСТ ПУСТОЙ”
==============================

Если в предоставленных документах нет ни одной релевантной новости по теме (или список источников пуст / все источники нерелевантны запросу),
то напиши ТОЛЬКО 3 блока:

### 1) Запрос и дата
### 2) Дайджест
### 3) Выводы и актуальность

В начале блоков 2 и 3 обязательно добавь:
“В данных нет новостей по теме, ниже — общий комментарий без опоры на источники”.

Контекст считается пустым, если выполнено хотя бы одно:
• после заголовка ИСТОЧНИКИ нет ни одного документа;
• или все документы нерелевантны запросу.

========================
ОГРАНИЧЕНИЕ ОБЪЁМА
==================

• Дайджест: 4–10 абзацев (можно более развернуто и литературно, но без потери точности).
• Выводы: 2–6 буллетов.
• Таймлайн: полный по всем датам из контекста (без лимита по числу строк).

========================
ТРЕБОВАНИЯ К СОДЕРЖАНИЮ БЛОКОВ
==============================

### 1) Запрос и дата
* Запрос: ...
* Опорная дата: ...

### 2) Дайджест (расширенный, “красивый” язык без потери строгости)
Требования:
• Начни с “за последнее время” (самые близкие к опорной дате документы): 2–5 ключевых фактов с датами и каналами.
• Затем опиши предысторию и “волны/разрывы” (если они видны).
• Встрой в текст мини-срез “как писали каналы” (без отдельного блока):
  - какие каналы что подчеркивают/какую рамку выбирают/какую лексику используют (нейтрально),
  - 2–4 опорные точки по каждому заметному каналу (даты в скобках допустимы для “упоминал/писал”, но не для одного числа).
• Если рядом с опорной датой нет материалов — явно напиши, что по данным сделать вывод о текущем положении нельзя.

### 3) Выводы и актуальность
2–6 буллетов, фокус на наблюдаемых сигналах:
* свежесть (дата последнего упоминания и разрыв до опорной даты, если можно посчитать)
* плотность (волна или единичные упоминания)
* ширина (сколько каналов)
* концентрация (доминирует ли один)
* новостность (похоже на разовое событие или на регулярный мониторинг/серии публикаций)
* ограничения данных (разрывы/шум/чего не хватает)
Запрещено: “важно/значимо” без опоры на сигналы выше.

### 4) Таймлайн
* Список строк, от нового к старому:
  “YYYY-MM-DD — Канал1; Канал2: кратко что произошло (по формулировке источника)”
* Если объединяешь каналы на одной дате — формулировка должна быть совместимой для всех перечисленных каналов.
  Если у каналов на одной дате разные конкретные значения/утверждения — перечисляй их через “;” в рамках одной строки, не усредняя и не подменяя.

========================
ФОРМАТ ВЫВОДА (СТРОГО)
======================

ЕСЛИ КОНТЕКСТ НЕ ПУСТОЙ:
### 1) Запрос и дата
...

### 2) Дайджест
...

### 3) Выводы и актуальность
* ...
* ...

### 4) Таймлайн
* YYYY-MM-DD — ...
* ...

ЕСЛИ КОНТЕКСТ ПУСТОЙ:
### 1) Запрос и дата
...

### 2) Дайджест
В данных нет новостей по теме, ниже — общий комментарий без опоры на источники.
...

### 3) Выводы и актуальность
В данных нет новостей по теме, ниже — общий комментарий без опоры на источники.
* ...
* ...
"""

        self.JUDGE_SYSTEM = """Ты — строгий эксперт по информационному поиску по новостям (в т.ч. экономическим).

Твоя задача: оценить релевантность кандидатной новости запросу. Запрос может быть:
- коротким топиком (например "курс рубля к доллару"),
- или текстом другой новости (тогда запрос описывает конкретный инфоповод).

Используй ТОЛЬКО текст кандидатного документа. Ничего не додумывай.

Шкала релевантности:
2 — документ явно про то же самое: отвечает топику ИЛИ описывает тот же инфоповод/факт/событие, что и запрос.
1 — документ связан по теме/контексту, но это немного другой инфоповод, или про то же, но без прямого соответствия.
0 — нерелевантно совсем.

Правило строгости:
ставь 2 только если связь очевидна по тексту документа; если информации недостаточно — ставь 0 или 1.

Верни строго валидный JSON и ничего больше:
{"relevance": 0|1|2}
"""

    @staticmethod
    def tokenize_ru(text: str):
        """Простая токенизация для BM25"""
        text = text.lower()
        text = re.sub(r"[^0-9a-zа-яё\s]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    @staticmethod
    def snippet(t: str, n: int = 1000) -> str:
        return t[:n]

    @staticmethod
    def _parse_relevance(text: str) -> int:
        """Парсинг JSON ответа от Judge LLM"""
        text = text.strip()
        # Пытаемся найти JSON блок
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            blob = m.group(0)
            try:
                obj = json.loads(blob)
                val = int(obj.get("relevance", 0))
                return val if val in (0, 1, 2) else 0
            except Exception:
                pass
        # Фоллбэк: ищем просто relevance: N
        m2 = re.search(r"relevance\"\s*:\s*([012])", text)
        if m2:
            return int(m2.group(1))
        return 0

    def _dense_search(self, query: str, top_k: int):
        """Векторный поиск через FAISS"""
        q_vec = self.encoder.encode(["query: " + query], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        scores, idx = self.index.search(q_vec, top_k)
        return idx[0].astype(int), scores[0].astype(np.float32)

    def _hybrid_retrieve_rrf(self, query: str, k: int = 50, topN_each: int = 500, k_rrf: int = 60, anchor_date=None) -> pd.DataFrame:
        """Гибридный поиск (Dense + BM25) с фильтрацией по дате ДО поиска в BM25"""
        
        # 1. Dense Search
        d_idx, d_sc = self._dense_search(query, topN_each)
        
        # Фильтрация по дате (если задана)
        if anchor_date is not None:
            ad = pd.to_datetime(anchor_date, utc=True).normalize()
            # Проверяем, есть ли колонка date_day или date
            date_col = "date_day" if "date_day" in self.df.columns else "date"
            dts = pd.to_datetime(self.df[date_col], errors="coerce", utc=True).dt.normalize()
            allowed_mask = (dts <= ad).to_numpy(dtype=bool)
            
            # Фильтруем dense результаты
            keep = allowed_mask[d_idx]
            d_idx = d_idx[keep]
        else:
            allowed_mask = None

        dense_rank = {int(rowpos): r for r, rowpos in enumerate(d_idx, start=1)}

        # 2. BM25 Search
        if self.bm25 is None:
            # Fallback to dense only
            out = self.df.iloc[d_idx].copy()
            out["score_rrf"] = [1.0 / (k_rrf + r) for r in range(1, len(d_idx) + 1)]
            return out.head(k).reset_index(drop=True)

        tokenized_query = self.tokenize_ru(query)
        bm_scores = self.bm25.get_scores(tokenized_query).astype(np.float32)
        
        # Зануляем скоры для дат из будущего ПЕРЕД сортировкой
        if allowed_mask is not None:
            bm_scores[~allowed_mask] = -np.inf

        # Берем топ-N индексов
        # Используем argpartition для скорости, потом сортируем
        if len(bm_scores) > topN_each:
            b_idx_unsorted = np.argpartition(-bm_scores, topN_each)[:topN_each]
            b_idx = b_idx_unsorted[np.argsort(-bm_scores[b_idx_unsorted])]
        else:
            b_idx = np.argsort(-bm_scores)[::-1]

        bm_rank = {int(rowpos): r for r, rowpos in enumerate(b_idx, start=1)}

        # 3. RRF Fusion
        union = np.array(sorted(set(dense_rank) | set(bm_rank)), dtype=int)
        rrf = np.zeros(len(union), dtype=np.float32)

        for j, rowpos in enumerate(union):
            if rowpos in dense_rank:
                rrf[j] += 1.0 / (k_rrf + dense_rank[rowpos])
            if rowpos in bm_rank:
                rrf[j] += 1.0 / (k_rrf + bm_rank[rowpos])

        # Сортировка по RRF
        order = np.argsort(-rrf)
        final_indices = union[order][:k]
        final_scores = rrf[order][:k]

        out = self.df.iloc[final_indices].copy()
        out["score_rrf"] = final_scores
        return out.reset_index(drop=True)

    def _judge_filter(self, cand: pd.DataFrame, query: str, threshold: int = 1, batch_size: int = 32) -> pd.DataFrame:
        """LLM-as-a-Judge фильтрация"""
        if cand.empty:
            return cand

        prompts = []
        for _, row in cand.iterrows():
            # Обрезаем текст для скорости (как в ноутбуке snip_chars=1200)
            doc = str(row["message"])[:1200]
            ch = str(row.get("channel_name", ""))
            dt = str(row.get("date_day", "")) or str(row.get("date", ""))

            user_msg = (
                f"ЗАПРОС:\n{query}\n\n"
                f"КАНДИДАТ:\n"
                f"channel={ch}\n"
                f"date={dt}\n"
                f"text:\n{doc}\n"
            )

            messages = [
                {"role": "system", "content": self.JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=40)
        
        # Batch generation
        outputs = self.llm.generate(prompts, sampling)
        
        relevances = []
        for o in outputs:
            txt = o.outputs[0].text
            relevances.append(self._parse_relevance(txt))

        out_df = cand.copy()
        out_df["judge_relevance"] = relevances
        
        return out_df[out_df["judge_relevance"] >= threshold].copy().reset_index(drop=True)

    def _build_context(self, query: str, cand: pd.DataFrame, anchor_date: str, snip_chars: int = 850) -> str:
        """Сборка контекста для финального промпта"""
        blocks = []
        for i, row in enumerate(cand.itertuples(index=False), start=1):
            # Пытаемся достать дату, если это Timestamp преобразуем
            d_val = getattr(row, "date_day", getattr(row, "date", ""))
            if isinstance(d_val, pd.Timestamp):
                date_day = d_val.strftime("%Y-%m-%d")
            else:
                date_day = str(d_val)[:10]

            channel = getattr(row, "channel_name", "Unknown")
            text = getattr(row, "message", "")

            blocks.append(f"[{i}] date={date_day} channel={channel}\n" + self.snippet(str(text), snip_chars))

        return (
            f"anchor_date: {anchor_date}\n"
            f"ВОПРОС/ЗАПРОС:\n{query}\n\n"
            f"ИСТОЧНИКИ:\n" + "\n\n".join(blocks)
        )

    def generate(self, query: str, anchor_date: str, 
                 k_retrieve: int = 50, 
                 k_docs_final: int = 25,
                 judge_threshold: int = 1):
        """
        Основной метод запуска пайплайна:
        1. Hybrid Retrieve
        2. Judge Filter (опционально, если threshold > 0)
        3. Generate Report
        """
        
        # 1. Retrieve
        print(f"Retrieving for query: {query}...")
        candidates = self._hybrid_retrieve_rrf(query, k=k_retrieve, anchor_date=anchor_date)
        
        # Сохраняем "грязных" кандидатов для дебага
        candidates_raw = candidates.copy()
        
        # 2. Judge Filter
        if judge_threshold > 0:
            print(f"Filtering with Judge (threshold={judge_threshold})...")
            candidates = self._judge_filter(candidates, query, threshold=judge_threshold)
            print(f"Kept {len(candidates)} / {len(candidates_raw)} docs.")
            
        # Лимитируем количество для контекста
        candidates_final = candidates.head(k_docs_final)
        
        # 3. Generate
        context = self._build_context(query, candidates_final, anchor_date)
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Параметры генерации из ноутбука (max_new_tokens=5000 в примере)
        sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=5000)
        
        print("Generating report...")
        request_output = self.llm.generate([prompt], sampling)[0]
        generated_text = request_output.outputs[0].text.strip()
        
        return {
            "summary": generated_text,
            "context": context,
            "candidates_raw": candidates_raw,
            "candidates_filtered": candidates
        }