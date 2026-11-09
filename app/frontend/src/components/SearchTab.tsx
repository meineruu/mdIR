import { useEffect, useRef, useState } from "react";
import { fetchTestQueries, postSearch, type SearchResponse, type TestQuery } from "../api";
import { fmt } from "../format";
import MetricsTable from "./MetricsTable";
import ResultColumn from "./ResultColumn";

const EXAMPLES = [
  { id: "PLAIN-2670", teks: "Is Caramel Color Carcinogenic?", status: "Berhasil" },
  { id: "PLAIN-2386", teks: "worms", status: "Berhasil" },
  { id: "PLAIN-3342", teks: "Is Coconut Milk Good For You?", status: "Qrels tidak lengkap" },
  { id: "PLAIN-133", teks: "Starving Tumors of Their Blood Supply", status: "Masih gagal" },
];

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

  return (
    <div className="space-y-5">
      <section className="space-y-4 rounded-xl bg-surface p-4 shadow-sm sm:p-5">
        <form
          className="flex flex-col gap-2 sm:flex-row"
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
            placeholder="Tulis query berbahasa Inggris, misalnya: vitamin D and cancer"
            className="min-w-0 flex-1 rounded-lg border border-line bg-white px-3 py-2.5 outline-none focus:border-brand focus:ring-2 focus:ring-brand/20"
          />
          <button
            type="submit"
            disabled={loading}
            className="rounded-lg bg-brand px-5 py-2.5 font-medium text-white hover:opacity-90 disabled:opacity-60"
          >
            {loading ? "Mencari…" : "Cari"}
          </button>
        </form>

        <div className="grid gap-3 md:grid-cols-[1fr_auto] md:items-center">
          <div>
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
              className="w-full rounded-lg border border-line bg-white px-3 py-2 text-sm"
            >
              <option value="">Ambil query dari split test ({testQueries.length || 323} query, bisa dibandingkan dengan ground truth)</option>
              {testQueries.map((q) => (
                <option key={q.id} value={q.id}>
                  {q.teks.length > 80 ? `${q.teks.slice(0, 80)}…` : q.teks} ({q.n_relevan} relevan)
                </option>
              ))}
            </select>
            {listError && <p className="mt-1 text-sm text-red-700">Daftar query gagal dimuat: {listError}</p>}
          </div>
          <label className="flex cursor-pointer items-center gap-2 text-sm text-ink-2">
            <input type="checkbox" checked={useCe} onChange={(e) => toggleCe(e.target.checked)} className="size-4 accent-[#0f5e6b]" />
            Pakai cross-encoder
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm text-ink-2">Contoh:</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex.id}
              onClick={() => pick(ex.id, ex.teks)}
              className={`rounded-full px-3 py-1 text-sm transition-colors ${
                testId === ex.id ? "bg-brand text-white" : "bg-page text-ink hover:bg-line"
              }`}
            >
              {ex.teks} <span className="opacity-70">· {ex.status}</span>
            </button>
          ))}
        </div>
      </section>

      {error && (
        <p role="alert" className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-800">
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

      {!data && !loading && !error && (
        <p className="rounded-xl bg-surface p-8 text-center text-ink-2">
          Pilih salah satu contoh di atas untuk melihat perbandingan BM25 dan Final stacking.
        </p>
      )}

      {data && (
        <div className={`space-y-5 transition-opacity ${loading ? "opacity-50" : ""}`} aria-busy={loading}>
          <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 text-sm text-ink-2">
            <span className="text-base font-medium text-ink">“{data.query}”</span>
            {data.split_test_id && <span>{data.split_test_id}</span>}
            {data.n_relevan !== null && <span>{data.n_relevan} dokumen relevan di qrels</span>}
            <span>{fmt(data.waktu_ms / 1000, 2)} detik</span>
          </div>

          {!data.pakai_cross_encoder && (
            <p className="rounded-lg bg-amber-50 px-4 py-2 text-sm text-amber-950">
              Cross-encoder dimatikan, jadi ini bukan konfigurasi resmi. Metriknya bisa berbeda dari model Final stacking.
            </p>
          )}

          {data.metrik ? (
            <MetricsTable bm25={data.metrik.bm25} final={data.metrik.final} />
          ) : (
            <p className="rounded-lg bg-surface px-4 py-3 text-sm text-ink-2">
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
