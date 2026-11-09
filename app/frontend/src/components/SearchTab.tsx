import { useEffect, useRef, useState } from "react";
import { fetchTestQueries, postSearch, type SearchResponse, type TestQuery } from "../api";
import { fmt } from "../format";
import MetricsTable from "./MetricsTable";
import PhoneMock from "./PhoneMock";
import ResultColumn from "./ResultColumn";

const EXAMPLES = [
  { id: "PLAIN-2670", teks: "Is Caramel Color Carcinogenic?", status: "Berhasil" },
  { id: "PLAIN-2386", teks: "worms", status: "Berhasil" },
  { id: "PLAIN-3342", teks: "Is Coconut Milk Good For You?", status: "Qrels tidak lengkap" },
  { id: "PLAIN-133", teks: "Starving Tumors of Their Blood Supply", status: "Masih gagal" },
];

const FACTS = ["3.633 dokumen NFCorpus", "323 query test", "Dicek dengan ground truth", "Kontribusi tiap sinyal"];

function Check() {
  return (
    <span className="grid size-5 place-items-center rounded-full bg-brand text-[11px] font-bold text-white" aria-hidden>
      ✓
    </span>
  );
}

export default function SearchTab() {
  const [text, setText] = useState("");
  const [testId, setTestId] = useState<string | null>(null);
  const [useCe, setUseCe] = useState(true);
  const [testQueries, setTestQueries] = useState<TestQuery[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  const [data, setData] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const pending = useRef<AbortController | null>(null);

  useEffect(() => {
    let active = true;
    let timer: number | undefined;
    // Backend butuh sekitar satu menit untuk memuat model, jadi daftar dicoba ulang sampai berhasil.
    const load = () =>
      fetchTestQueries()
        .then((rows) => {
          if (!active) return;
          setTestQueries(rows);
          setListError(null);
        })
        .catch((e: Error) => {
          if (!active) return;
          setListError(e.message);
          timer = window.setTimeout(load, 3000);
        });
    load();
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, []);

  async function run(query: string, id: string | null, ce: boolean) {
    if (!id && !query.trim()) {
      setError("Tulis query dulu, atau pilih query dari split test.");
      return;
    }
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    setLoading(true);
    setError(null);
    try {
      setData(await postSearch({ query, pakai_cross_encoder: ce, split_test_id: id }, controller.signal));
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      setError((e as Error).message);
    }
    if (pending.current === controller) setLoading(false);
  }

  function pick(id: string, teks: string) {
    setTestId(id);
    setText(teks);
    run(teks, id, useCe);
  }

  function toggleCe(next: boolean) {
    setUseCe(next);
    if (data) run(text, testId, next);
  }

  function reset() {
    pending.current?.abort();
    setLoading(false);
    setData(null);
    setError(null);
    setText("");
    setTestId(null);
  }

  const hero = !data;

  return (
    <div className="space-y-5">
      <section className={hero ? "grid grid-cols-1 items-center gap-8 pt-4 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,0.75fr)]" : ""}>
        <div className={hero ? "space-y-6" : "space-y-4 rounded-xl bg-surface p-4 shadow-sm sm:p-5"}>
          {hero && (
            <>
              <h2 className="text-4xl leading-[1.12] font-normal sm:text-5xl lg:text-[3.4rem]">
                <span className="underline decoration-ink decoration-dotted decoration-2 underline-offset-[6px]">Literatur medis</span>
                <br />
                yang relevan, dari
                <br />
                bahasa sehari-hari
              </h2>
              <p className="text-lg text-ink-2">
                <strong className="font-semibold text-ink">Final stacking</strong> mencapai P@5 0,437 pada split test, naik dari 0,309 (BM25)
              </p>
              <ul className="flex flex-wrap gap-2">
                {FACTS.map((f) => (
                  <li key={f} className="flex items-center gap-2 rounded-full border border-line bg-white px-3 py-1.5 text-sm">
                    <Check /> {f}
                  </li>
                ))}
              </ul>
            </>
          )}

          <form
            className="flex items-center gap-2 rounded-full border border-line bg-white p-1.5 shadow-sm focus-within:border-brand"
            onSubmit={(e) => {
              e.preventDefault();
              run(text, testId, useCe);
            }}
          >
            <label className="sr-only" htmlFor="q">
              Query pencarian
            </label>
            <input
              id="q"
              value={text}
              onChange={(e) => {
                setText(e.target.value);
                setTestId(null);
              }}
              placeholder="Tulis query berbahasa Inggris, misalnya: vitamin D"
              className="min-w-0 flex-1 bg-transparent px-4 py-2 outline-none"
            />
            <button type="submit" disabled={loading} className="btn-glow shrink-0 rounded-full bg-brand px-6 py-2.5 font-medium text-white hover:opacity-90 disabled:opacity-60">
              {loading ? "Mencari…" : "Cari"}
            </button>
          </form>

          <div className="grid gap-3 md:grid-cols-[1fr_auto] md:items-center">
            <div className="min-w-0">
              <label className="sr-only" htmlFor="test-query">
                Ambil query dari split test
              </label>
              <select
                id="test-query"
                value={testId ?? ""}
                onChange={(e) => {
                  const row = testQueries.find((q) => q.id === e.target.value);
                  if (row) pick(row.id, row.teks);
                }}
                className="w-full rounded-full border border-line bg-white px-4 py-2 text-sm"
              >
                <option value="">Ambil query dari split test ({testQueries.length || 323} query)</option>
                {testQueries.map((q) => (
                  <option key={q.id} value={q.id}>
                    {q.teks.length > 80 ? `${q.teks.slice(0, 80)}…` : q.teks} ({q.n_relevan} relevan)
                  </option>
                ))}
              </select>
              {listError && <p className="mt-1 text-sm text-red-700">Daftar query belum termuat, mencoba lagi: {listError}</p>}
            </div>
            <label className="flex cursor-pointer items-center gap-2 text-sm text-ink-2">
              <input type="checkbox" checked={useCe} onChange={(e) => toggleCe(e.target.checked)} className="size-4 accent-[#0a6cff]" />
              Pakai cross-encoder
            </label>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm text-ink-2">Contoh:</span>
            {EXAMPLES.map((ex) => (
              <button
                key={ex.id}
                onClick={() => pick(ex.id, ex.teks)}
                className={`rounded-full border px-3 py-1 text-sm transition-colors ${
                  testId === ex.id ? "border-brand bg-brand text-white" : "border-line bg-white text-ink hover:border-brand"
                }`}
              >
                {ex.teks} <span className="opacity-70">· {ex.status}</span>
              </button>
            ))}
          </div>
        </div>
        {hero && <PhoneMock />}
      </section>

      {error && (
        <p role="alert" className="rounded-xl bg-red-50 px-4 py-3 text-sm text-red-800">
          {error}
        </p>
      )}

      {loading && !data && (
        <div className="grid gap-4 lg:grid-cols-2" aria-label="Memuat hasil">
          {[0, 1].map((i) => (
            <div key={i} className="h-96 animate-pulse rounded-xl bg-line/60" />
          ))}
        </div>
      )}

      {data && (
        <div className={`space-y-5 transition-opacity ${loading ? "opacity-50" : ""}`} aria-busy={loading}>
          <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 text-sm text-ink-2">
            <span className="font-serif text-xl font-semibold text-ink">“{data.query}”</span>
            {data.split_test_id && <span>{data.split_test_id}</span>}
            {data.n_relevan !== null && <span>{data.n_relevan} dokumen relevan di qrels</span>}
            <span>{fmt(data.waktu_ms / 1000, 2)} detik</span>
            <button onClick={reset} className="font-medium text-brand hover:underline">
              Kembali ke awal
            </button>
          </div>

          {!data.pakai_cross_encoder && (
            <p className="rounded-xl bg-amber-50 px-4 py-2 text-sm text-amber-950">
              Cross-encoder dimatikan, jadi ini bukan konfigurasi resmi. Metriknya bisa berbeda dari model Final stacking.
            </p>
          )}

          {data.metrik ? (
            <MetricsTable bm25={data.metrik.bm25} final={data.metrik.final} />
          ) : (
            <p className="rounded-xl bg-surface px-4 py-3 text-sm text-ink-2 shadow-sm">
              Query bebas tidak punya ground truth, jadi status relevansi dan metrik tidak ditampilkan. Pilih query dari
              split test untuk membandingkannya dengan qrels.
            </p>
          )}

          <div className="grid items-start gap-4 lg:grid-cols-2">
            <ResultColumn title="BM25 (baseline)" results={data.bm25} final={false} />
            <ResultColumn title="Final stacking" results={data.final} final />
          </div>
        </div>
      )}
    </div>
  );
}
