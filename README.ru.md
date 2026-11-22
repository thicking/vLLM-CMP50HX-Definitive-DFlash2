<!-- Документация производной сборки; перечень изменений — PATCHES.md. -->
# vLLM-CMP50HX-Definitive-DFlash2

Язык: [English](README.md) | Русский

Самостоятельно сопровождаемая производная сборка [vLLM](https://github.com/vllm-project/vllm) и [weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive). В этом снимке сохранены восстановленный Experimental B runtime и downstream-изменения GP-12, GP-20, GP-21.1 и GP-21.3 для SM75/CMP50HX, а также соответствующие регрессионные тесты.

Это не новый inference engine. Авторы vLLM, weicj и авторы сторонних компонентов сохраняют авторство своей работы. См. [LICENSE](LICENSE), [NOTICE](NOTICE) и [PATCHES.md](PATCHES.md).

## Статус проекта

Последний релиз производного проекта — [`v0.2.0-alpha.2`](https://github.com/thicking/vLLM-CMP50HX-Definitive-DFlash2/releases/tag/v0.2.0-alpha.2), документальное дополнение без изменений runtime. Релиз исходников runtime остаётся [`v0.2.0-alpha.1`](https://github.com/thicking/vLLM-CMP50HX-Definitive-DFlash2/releases/tag/v0.2.0-alpha.1) **Workload-aware KV Cache & Parallel Inference**; унаследованная версия upstream fork — `v0.2.2-post2`. В alpha.2 не менялись код движка, профиль, launcher, модель или параметры production. Готовые wheel и container image не публикуются. Чистая сборка и установка непосредственно из этого репозитория на CMP50HX не проверялись.

Материалы GP-24/GP-25: [руководство Prefix Cache](docs/prefix-cache.md), [измерения](docs/benchmarks.md), [подробный Lessons Learned на русском](docs/lessons-learned.ru.md).

Первоначальный снимок Experimental B основан на upstream-fork commit `2d63e2e322b7488776d1fb822f3556dbd7334116`. Исходное дерево было shallow и dirty; локальные изменения импортированы одним snapshot-коммитом, без искусственной истории патчей. Последующие коммиты GP-20, GP-21.1 и GP-21.3 сохранены в истории релиза. Полное дерево исходников на момент перезагрузок сервера 10 ноября 2026 года отсутствует, поэтому наличие каждого восстановленного изменения в тот момент — **UNKNOWN**.

## Оборудование и проверенная конфигурация runtime

Проверенная текущая конфигурация: **2 × NVIDIA CMP 50HX 20 GB**, Turing TU102 / SM75, TP=2 на GPU0/1.

| Параметр | Проверенная конфигурация GP-21.3 |
| --- | --- |
| Target | Swift Qwen3.8-27B AWQ, `compressed-tensors` |
| Draft | DFlash2 GPTQ INT8, TP=2 |
| Target KV | FP8 |
| Draft KV | FP16 |
| Лимит контекста | 221 184 токена на запрос |
| Лимит scheduler | 2 последовательности |
| Speculative tokens | 7 |
| CUDA Graphs | `FULL_AND_PIECEWISE`, размеры capture 8 и 16; DFlash2 full graphs включены |
| Prefix cache | Включён; Mamba cache mode `align` |
| Custom AllReduce | Выключен |
| Целевое GPU utilization при старте | 0.986 |
| Max batched tokens | 4 096 |

После GP-21.3 лог allocator сообщил KV capacity 247 398 токенов (около 1,12× настроенного контекста одного запроса; ранее было 252 313). KV распределяется между запросами динамически. Это **не** означает, что два запроса могут одновременно занять по 221 184 токена каждый. `MAX_NUM_SEQS=2` разрешает два scheduler slot; параллельный запуск также зависит от формы запросов и независимых GDN/Mamba state pools.

Переносимый route preset: [`profiles/2xCMP50HX/qwen27b/w4a16/dflash2-fp8kv-2x221K-text-image.env`](profiles/2xCMP50HX/qwen27b/w4a16/dflash2-fp8kv-2x221K-text-image.env). В нём только параметры route. Пути к target/draft checkpoint, выбор GPU, сеть и cache directories задавайте через окружение существующего launcher. До старта просмотрите итоговую конфигурацию: `./launcher.sh --print-config`.

Профиль отражает восстановленную конфигурацию B. Это не обещание двух полных контекстов и не рекомендация для любых Turing-карт. В репозитории нет весов модели и приватных launcher-файлов сервера.

## Изменения runtime

- **Scheduler:** активный decode больше не задерживается только из-за независимого prefill; добавлена диагностика admission, running/waiting и отказов по capacity.
- **GDN/Mamba reservation:** резерв учитывает параллельные DFlash/DSpark запросы, resident state, verifier pages, prefill checkpoints и copy-on-write при prefix-cache hit.
- **Prefix lifecycle:** hashed align-mode state blocks сохраняются до освобождения владельцем.
- **DFlash2 restored prefix:** длина prefix, восстановленного только в target KV, сохраняется; каждая свежесобранная draft block table сдвигается один раз за scheduler step. При новом admission slot переинициализируется. CPU regression test проверяет restore, следующий шаг, logical/physical mapping, preemption/resume, повторное использование slot и обычный decode.
- **KV allocator:** добавлена диагностика required/free/capacity/watermark/reservation; ускорение allocator не заявляется.
- **Streaming parser:** `IndexError` для незавершённого partial-JSON фрагмента считается неполным префиксом потока.
- **Qwen3 tool parser:** добавлена поддержка структурированного required/named tool-choice пути. Сочетание reasoning с `tool_choice=required` остаётся нестабильным и не считается исправленным.
- **SM75 DFlash2:** сохранены восстановленные GPTQ/Marlin projection paths и Turing-специфичные DFlash2 kernels. Часть этих оптимизаций появилась до GP-12; происхождение описано в [PATCHES.md](PATCHES.md).

## Изменения v0.2.0-alpha.1

- **Live prefill telemetry (GP-20):** добавлен накопительный счётчик, экспортируемый как `vllm:prefill_computed_tokens_total`, а также `vllm:num_requests_prefilling` и `vllm:prefill_tokens_remaining`. Счётчик растёт только после успешно завершённых локальных prefill-операций. Prefix-cache и external-KV токены, decode и speculative verification не считаются; фактически повторно вычисленные токены после preemption учитываются. Семантика прежних prompt-token counters не менялась.
- **Корректность prefill barrier (GP-21.1):** более поздний waiter, который пока нельзя принять, больше не расширяет alignment frontier после отказа KV allocator для более ранней FIFO-заявки. Активная prefill cohort может продолжить обработку; отказанный head повторяется после завершения cohort. Исправление сохраняет prefill barrier и `full_sequence_must_fit`.
- **Ограниченная scheduler-диагностика (GP-21.1):** ограничивает объём per-request состояния, подавляет неизменные повторы с backoff и сразу показывает изменения состояния и восстановление. Решения scheduler не меняются.
- **Workload-aware DFlash2 pools (GP-21.3):** добавлен opt-in параметр `dflash_draft_pool_concurrency`. При его включении allocator использует фактические cache specs, физические размеры страниц, доступный объём памяти и заданную concurrency draft, сохраняя резервы target Mamba. Значение `None` сохраняет прежнее пропорциональное распределение.

### Параметр draft pool

Точный CLI-параметр vLLM: `--dflash-draft-pool-concurrency N`; имя параметра в профиле/launcher — `DFLASH_DRAFT_POOL_CONCURRENCY=N`. `N` — целое число ≥1 и не больше scheduler `max_num_seqs`. Настройка применяется только для native DFlash2 с независимыми KV pools и определяемыми группами DFlash `SlidingWindowSpec`. Если параметр не задан, используется прежнее распределение. Явный пользовательский `num_gpu_blocks_override` несовместим с workload-aware планом. Проверка выполняется при реальном sizing cache; временный внутренний override для CUDA Graph profiling поддерживается.

Для уже настроенной native DFlash2 команды `vllm serve` добавьте:

```bash
--dflash-draft-pool-concurrency 2
```

Соответствующая форма profile:

```dotenv
MAX_NUM_SEQS=2
DFLASH_DRAFT_POOL_CONCURRENCY=2
```

Параметр задаёт sizing draft pools, а не увеличивает лимит запросов scheduler. Распределение зависит от модели, cache layout и оборудования. Скалярный KV headline не описывает независимые capacities, размеры страниц, выравнивание и резервы hybrid Mamba/attention/draft groups. Вектор CMP50HX ниже относится только к проверенному маршруту и не является универсальным preset.

### GPU acceptance GP-21.3

На проверенном маршруте с двумя CMP50HX распределение cache groups изменилось с `[23,23,23,154,10,10,10,10,10]` на `[23,23,23,151,18,18,18,18,18]`. Ёмкость каждого из пяти draft pools выросла на 80% (10→18 блоков); target full-attention capacity уменьшилась примерно на 1,95% (154→151). Настроенный контекст одного запроса остался 221184. Скалярная KV capacity в логах изменилась с 252313 до 247398 токенов (−4915, около −1,95%); это не гарантирует одновременное размещение двух полностью заполненных контекстов.

| Параллельные prompts, токены | TTFT каждого запроса |
| --- | --- |
| 20 971 + 20 968 | 45,597 / 46,843 с |
| 32 012 + 32 011 | 70,37 / 71,22 с |
| 64 004 + 64 003 | 147,44 / 149,35 с |
| 95 996 + 95 995 | 245,92 / 246,69 с |
| 199 990 + 16 000 | 346,38 / 344,96 с |

Пара около 21K сгенерировала 1353 / 1494 токена со скоростью 58,34 / 59,47 ток/с на запрос. Сумма индивидуальных скоростей — 117,81 ток/с — **не** является отдельно измеренной aggregate throughput. В остальных длинных сценариях генерировалось по два токена на запрос; это проверка admission/prefill, а не полезной длительной генерации. В baseline два запроса примерно по 21K получили TTFT 22,9 / 44,4 с при `running=1`, `waiting=1` и последовательном prefill. Candidate подтверждает перекрытие двух запросов, но TTFT первого запроса выше в этом сравнении; ускорение prefill или общий сопоставимый прирост decode throughput не доказаны. Идентичный повтор запроса длиной 7980 токенов показал 4944 cached tokens. Полные условия и ограничения приведены в [TEST-REPORT.md](TEST-REPORT.md).

### Причина scheduler livelock

Сокращённый CPU reproducer выявил livelock FIFO/frontier: allocator отклоняет head очереди, blocked-marker исключает его из alignment frontier, более поздний недоступный waiter всё ещё влияет на frontier, активный prefill не продвигается, и scheduler возвращает пустой batch. Regression воспроизводился на исходниках Definitive v0.2.2, post2 и post3. Патч запрещает продвигать frontier за отклонённый head, пока работает активная cohort; barrier и `full_sequence_must_fit` не отключаются. Этот дефект отличается от более раннего Mamba-alignment livelock fix в upstream fork. Предложенный патч и текст issue находятся в [`upstream-proposals/`](upstream-proposals/); они ещё не отправлялись.

CUDA Graphs, DFlash2=7, target/draft quantization, TP=2 и parser configuration сохранены. Production graph capture проходил при запуске восстановленной B; clean checkout ещё не собирался, поэтому воспроизводимость graph capture из опубликованного исходника пока не подтверждена.

## Сборка и установка

На сервере, где работал runtime, были зафиксированы Python 3.12.14, PyTorch 2.13.0+cu130, Triton 3.7.1, FlashInfer 0.6.18.post1, `compressed-tensors` 0.17.0, Transformers 5.17.0 и driver 610.43.03. Это наблюдения одной установленной среды, а не полный lockfile. В production virtualenv отсутствовал модуль `pip`; фактические runtime shared libraries требуется подтвердить отдельно. Версия CUDA toolkit `nvcc` и чистая нативная сборка **NOT VERIFIED**.

В исходниках сохранён build entrypoint `./build.sh`; он ожидает согласованные зависимости и toolchain из `PROJECT_RELEASE.env` и `requirements/build/`. Сборка из нового checkout не подтверждена. Используйте отдельную машину или изолированную build-среду и сохраняйте полный build log. Не копируйте сюда production virtualenv и скомпилированные cache.

Базовые проверки исходников:

```bash
bash -n build.sh launcher.sh tools/validate_profiles.sh
bash tools/validate_profiles.sh
python3 -m py_compile vllm/v1/core/sched/scheduler.py \
  vllm/v1/kv_cache_interface.py \
  vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
```

Profile validator проверяет структуру route-файлов, но не capacity GPU и не запуск runtime. GPU regression suite ещё неполный.

## Runtime-проверки на CMP50HX

Experimental B фактически испытывался на сервере с двумя CMP50HX; проверка не ограничивалась CPU-тестами или анализом исходников. GP-12 включал использование production runtime и ограниченные GPU acceptance-сценарии. Ниже указаны конкретные условия: это доказательство проверенных путей, а не гарантия для всех нагрузок.

- **10-минутный поток и параллельный запрос (2026-11-14):** при лимите 147 456 токенов на запрос поток вернул HTTP 200 и продолжался до клиентского ограничения 600 секунд после первого токена (TTFT 1,31 с; 12 513 chunks, 240 939 символов). Во время потока неравномерный запрос с prompt 2 364 токена и output 17 токенов вернул HTTP 200 за 3,46 с. В одном scheduler batch зафиксированы оба ID и `running=2` — это подтверждает фактическое перекрытие именно этой пары. DFlash-счётчики достигли 13 025 draft / 33 161 accepted. KV capacity в логах — 229 376 токенов. Проверенная пара динамически поместилась; два запроса, каждый полностью заполняющий свой лимит, не проверялись и превысили бы эту capacity. 147 456 — наибольший проверенный контекст для этой пары; тестовая пара на 160K не поместилась. Это не универсальный максимум.
- **Проверка текущего профиля после reboot (2026-11-15):** B запущен на двух GPU с контекстом 221 184, `MAX_NUM_SEQS=2`, DFlash2=7, prefix caching, CUDA Graphs и `qwen3_xml`. Одновременные text-stream и `lookup_city` tool-запрос вернули HTTP 200; scheduler показал `running=2`, JSON инструмента был корректным, DFlash-счётчики выросли. Повтор одного prompt длиной 20 925 токенов с named tool завершился трижды (25,793 / 4,541 / 3,343 с); prefix hits выросли 0 → 18 128 → 36 256. Это подтверждает проверенный GPU-маршрут восстановления prefix и повторного admission, но не все варианты partial block или preemption. Также прошёл неравномерный сценарий: около 190 126 content-токенов плюс короткий tool-запрос (пик KV 77,27%; точный полный chat-template token count не записан).
- **Агентная endurance-проверка (2026-11-15):** изолированная coding-agent задача завершила 28 циклов, 84 tool-вызова, 28 изменений и 56 чтений за 1 828,9 с (~30,5 минуты). Health оставался true, PID сервиса не менялся, ошибок guard/runtime не зафиксировано. В 177 сэмплах GPU0/1 достигали 100% загрузки, максимальная температура — 60/69 °C; GPU2 оставался idle. Минимальные `MemAvailable` и `SwapFree` составили 5,601 и 7,523 GiB; memory PSI `avg10` не превышал 0. Это ограниченный прогон, не многочасовая или многодневная endurance-проверка.
- **Cancellation и клиенты:** cancellation проверялся в GP-12; зависания runtime или отказа сервиса не отмечено. С B также проверялись реальные клиентские сценарии OpenCode и Open WebUI. Это подтверждённые владельцем acceptance-наблюдения, но не полная матрица совместимости версий; время cancellation и версии UI не записаны в опубликованных тестовых артефактах.
- **Текущая эксплуатация:** на 2026-11-15 production `vllm-b.service` работал на GPU0/1; `/health` отвечал 200, а `/v1/models` показывал `swift15-qwen38-awq` и контекст 221 184. Владелец сообщает о средней скорости около **90 ток/с** на реальных задачах и стабильной работе. Это предварительное наблюдение текущего использования, не контролируемый benchmark: нет фиксированных prompt, контекста, concurrency и окна измерения.

В отдельном B-only throughput snapshot один cold-prefix output длиной 256 токенов показал около 41,76 ток/с; два одновременных output по 128 токенов — оценочно 52,85 / 49,55 ток/с на запрос. Эти краткие оценки относятся к конкретным условиям и напрямую несопоставимы с пользовательским наблюдением ~90 ток/с. Сравнение с Neith или заявление об общем ускорении не делается.

## Чистая сборка и оставшиеся GPU-проверки

Runtime-результаты выше получены на восстановленной рабочей копии Experimental B. **Чистая нативная сборка и установка непосредственно из опубликованного GitHub checkout не воспроизведены.** Это отдельный вопрос сборочной воспроизводимости; он не отменяет испытания восстановленной сборки. Wheel и container image не публиковались.

Capture CUDA Graphs наблюдался при старте, а CUDA Graphs оставались активны во время runtime acceptance. **Replay CUDA Graph после повторного использования slot** ещё не проверен отдельно. Полный GDN/Mamba slot lifecycle и GPU preemption/resume остаются неполными; проверка cancellation не означает проверку preemption/resume. CPU cache-shift regression воспроизвёл FAIL исходного варианта и PASS после патча; GPU prefix restore был проверен на повторном prompt, но это не покрывает все ветви GPU cache-shift lifecycle. Required tool calling при включённом reasoning также остаётся известным ограничением parser.

## Известные ограничения

- Причина трёх сообщённых перезагрузок хоста 10 ноября 2026 года остаётся **UNKNOWN**. Наблюдались RAM/swap pressure, но причинная связь не установлена.
- Нет доказательств, что DFlash2 cache-shift defect вызывал reboot; наличие дефекта в исходниках на момент reboot также **UNKNOWN**.
- Проверка реального scheduler preemption/resume неполная.
- Полный GDN slot lifecycle и CUDA Graph replay после повторного использования slot напрямую не подтверждены end-to-end.
- `tool_choice=required` вместе с reasoning остаётся нестабильным.
- Оставшиеся GPU-проверки неполны; опубликованный cache-shift lifecycle regression выполняется на CPU, хотя отдельный маршрут повторного GPU prefix restore был проверен.
- Чистая сборка из этого GitHub checkout не проверена.
- Значение ~90 ток/с — эксплуатационное наблюдение владельца, не формальный benchmark; сопоставимое сравнение с Neith отсутствует.
- Результаты относятся только к указанной конфигурации оборудования и программ.

## Upstream, лицензия и благодарности

Производная сохраняет структуру исходников vLLM и историю upstream fork. Авторы vLLM, weicj и сторонних компонентов сохраняют авторство своей работы. Существующие upstream acknowledgements и copyright notices сохранены.

Используемые или упоминаемые проекты: [vLLM](https://github.com/vllm-project/vllm), [vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [FlashQLA](https://github.com/QwenLM/FlashQLA), Marlin, Triton и авторы DFlash/DFlash2. Для их кода применяются соответствующие лицензии и notices. Unlock для GPU сопровождается авторами [xrip/cmp50hx-unlock](https://github.com/xrip/cmp50hx-unlock); модификация видеокарт и driver unlock не являются разработкой этого проекта.

Основная лицензия проекта — Apache-2.0. При распространении сохраняйте copyright, NOTICE, SPDX и сторонние лицензионные файлы. Проект сопровождается независимо и не заявляет права на исходный vLLM или сторонний код.

См. [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md), [CHANGELOG.md](CHANGELOG.md) и [PATCHES.md](PATCHES.md). Изменения upstream рассматриваются вручную; автоматического слияния в `main` нет.
