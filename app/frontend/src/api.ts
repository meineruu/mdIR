export interface TestQuery {
  id: string;
  teks: string;
  n_relevan: number;
}

export interface Sinyal {
  kode: string;
  nama: string;
  bobot: number;
  skor_ternormalisasi: number;
  kontribusi: number;
}

export interface Hasil {
  peringkat: number;
  doc_id: string;
  judul: string;
  cuplikan: string;
  skor: number;
  memuat_kata_query: boolean;
  kata_cocok: string[];
  kata_tampil: string[];
  punya_label_train: boolean;
  relevansi: number | null;
  sinyal: Sinyal[] | null;
}

export interface Metrik {
  p5: number;
  p10: number;
  ndcg10: number;
}

export interface SearchResponse {
  query: string;
  split_test_id: string | null;
  pakai_cross_encoder: boolean;
  kata_query: string[];
  n_relevan: number | null;
  waktu_ms: number;
  bm25: Hasil[];
  final: Hasil[];
  metrik: { bm25: Metrik; final: Metrik } | null;
}

export interface Delta {
  mean: number;
  ci95: [number, number];
  wilcoxon_p: number;
  better: number;
  worse: number;
  same: number;
}

export interface Summary {
  final_table: Record<string, { dev: Record<string, number>; test: Record<string, number> }>;
  error_analysis: Record<string, unknown>;
  final_vs_expansion: { selisih_p5: number; p_wilcoxon: number; naik: number; turun: number; sama: number };
}

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new Error("Tidak bisa menghubungi server. Pastikan backend sudah berjalan.");
  }
  if (res.status >= 500) {
    throw new Error("Backend belum siap atau sedang bermasalah. Tunggu sekitar satu menit setelah backend dinyalakan, lalu coba lagi.");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : `Permintaan gagal (kode ${res.status}).`;
    throw new Error(detail);
  }
  return res.json();
}

export const fetchTestQueries = () => call<TestQuery[]>("/api/queries/test");
export const fetchSummary = () => call<Summary>("/api/summary");

export const postSearch = (
  body: { query: string; pakai_cross_encoder: boolean; split_test_id: string | null },
  signal: AbortSignal,
) =>
  call<SearchResponse>("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
