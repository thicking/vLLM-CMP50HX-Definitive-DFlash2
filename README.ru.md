<!-- Документация производной сборки; перечень изменений — PATCHES.md. -->
# vLLM-CMP50HX-Definitive-DFlash2

Язык: [English](README.md) | Русский

Самостоятельно сопровождаемая производная сборка [vLLM](https://github.com/vllm-project/vllm) и [weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive). В этом снимке сохранены восстановленный Experimental B runtime и изменения GP-12 для scheduler, KV/GDN состояния, DFlash2, parser и регрессионных тестов.

Это не новый inference engine. Авторы vLLM, weicj и авторы сторонних компонентов сохраняют авторство своей работы. См. [LICENSE](LICENSE), [NOTICE](NOTICE) и [PATCHES.md](PATCHES.md).

## Статус проекта

Это экспериментальный релиз исходников `v0.1.0-alpha.1`. Git-тег проекта не зависит от версии Python-пакета vLLM (`0.2.3.dev0+g2d63e2e32.d20260928.cu130`) и upstream fork (`v0.2.2-post2`). Готовые wheel и container image не публикуются. Сборка из чистого checkout ещё не проверялась на CMP50HX.

Импорт соответствует восстановленному Experimental B на базе upstream-fork commit `2d63e2e322b7488776d1fb822f3556dbd7334116`. Исходное дерево было shallow и dirty; локальные изменения импортированы одним snapshot-коммитом, без искусственной истории патчей. Полное дерево исходников на момент перезагрузок сервера 10 ноября 2026 года отсутствует, поэтому присутствие всех восстановленных изменений во время этих событий — **UNKNOWN**.

## Оборудование и проверенная конфигурация runtime

Проверенная текущая конфигурация: **2 × NVIDIA CMP 50HX 20 GB**, Turing TU102 / SM75, TP=2 на GPU0/1.

| Параметр | Конфигурация восстановленного Experimental B |
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

Логи запуска этой конфигурации сообщали KV capacity 252 313 токенов (около 1,14× настроенного контекста одного запроса). KV распределяется между запросами динамически. Это **не** означает, что два запроса могут одновременно занять по 221 184 токена каждый. `MAX_NUM_SEQS=2` разрешает два scheduler slot; параллельный запуск также зависит от формы запросов и независимых GDN/Mamba state pools.

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

CUDA Graphs, DFlash2=7, target/draft quantization, TP=2 и parser configuration сохранены. Чистая сборка ещё не воспроизвела production graph capture.

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

## Проверки и производительность

Восстановленные данные включают два запроса, одновременно наблюдавшиеся в `running=2`, параллельные короткие streaming- и tool-запросы, повторное использование prefix и один изолированный 30-минутный coding-agent прогон (28 завершённых циклов, 84 вызова инструментов) без зарегистрированной runtime ошибки или перезапуска сервиса. Это ограниченные acceptance-наблюдения, а не гарантия endurance.

В этом релизе не публикуются PP/TG benchmark числа. Перед публикацией требуется независимо проверить происхождение и семантику метрик throughput из LLM Monitor; стандартизованного benchmark чистого checkout пока нет. Не сравнивайте эти наблюдения с несопоставимыми цифрами для одного запроса.

## Известные ограничения

- Причина трёх сообщённых перезагрузок хоста 10 ноября 2026 года остаётся **UNKNOWN**. Наблюдались RAM/swap pressure, но причинная связь не установлена.
- Нет доказательств, что DFlash2 cache-shift defect вызывал reboot; наличие дефекта в исходниках на момент reboot также **UNKNOWN**.
- Проверка реального scheduler preemption/resume неполная.
- Полный GDN slot lifecycle и CUDA Graph replay после повторного использования slot напрямую не подтверждены end-to-end.
- `tool_choice=required` вместе с reasoning остаётся нестабильным.
- GPU regression coverage неполный; опубликованный cache-shift lifecycle тест работает только на CPU.
- Чистая сборка не проверялась на GPU.
- Результаты относятся только к указанной конфигурации оборудования и программ.

## Upstream, лицензия и благодарности

Производная сохраняет структуру исходников vLLM и историю upstream fork. Авторы vLLM, weicj и сторонних компонентов сохраняют авторство своей работы. Существующие upstream acknowledgements и copyright notices сохранены.

Используемые или упоминаемые проекты: [vLLM](https://github.com/vllm-project/vllm), [vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [FlashQLA](https://github.com/QwenLM/FlashQLA), Marlin, Triton и авторы DFlash/DFlash2. Для их кода применяются соответствующие лицензии и notices. Unlock для GPU сопровождается авторами [xrip/cmp50hx-unlock](https://github.com/xrip/cmp50hx-unlock); модификация видеокарт и driver unlock не являются разработкой этого проекта.

Основная лицензия проекта — Apache-2.0. При распространении сохраняйте copyright, NOTICE, SPDX и сторонние лицензионные файлы. Проект сопровождается независимо и не заявляет права на исходный vLLM или сторонний код.

См. [CONTRIBUTING.md](CONTRIBUTING.md), [AGENTS.md](AGENTS.md), [CHANGELOG.md](CHANGELOG.md) и [PATCHES.md](PATCHES.md). Изменения upstream рассматриваются вручную; автоматического слияния в `main` нет.
