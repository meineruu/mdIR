# Deploy demo MedIR

Frontend di **Vercel**, backend di **Hugging Face Spaces**. Backend tidak bisa ditaruh di Vercel karena memakai PyTorch dan tiga model (sekitar 650 MB), jauh di atas batas ukuran fungsi Vercel.

Alurnya: browser membuka situs Vercel, lalu Vercel meneruskan setiap permintaan `/api/...` ke Space. Karena diteruskan dari server, tidak ada masalah CORS.

## 1. Backend di Hugging Face Spaces

1. Buat akun di huggingface.co, lalu buat token **Write** di Settings → Access Tokens.
2. Buat Space baru di huggingface.co/new-space:
   - Space name: `medir-api`
   - SDK: **Docker**, template **Blank**
   - Hardware: **CPU basic** (gratis)
   - Visibility: **Public**. Space privat tidak bisa diakses Vercel tanpa token. Konsekuensinya, kode backend, cache hasil, dan dataset di Space bisa dilihat publik.
3. Dari root repo ini, susun folder Space lalu unggah (ganti `USERNAME` dengan username Hugging Face-mu):

```bash
python deploy/hf-space/prepare_space.py ../medir-space
```

```bash
hf auth login
```

```bash
hf upload USERNAME/medir-api ../medir-space . --repo-type space
```

4. Tunggu build selesai di tab **Logs** Space (sekitar 10–15 menit untuk pertama kali, lalu sekitar 1 menit memuat model).
5. Cek `https://USERNAME-medir-api.hf.space/api/health`. Kalau muncul `{"status":"siap"}`, backend sudah jalan.

Space gratis tertidur setelah lama tidak dipakai. Buka URL health di atas beberapa menit sebelum presentasi supaya sudah bangun.

## 2. Frontend di Vercel

1. Di `app/frontend/vercel.json`, ganti `USERNAME` dengan username Hugging Face-mu, lalu commit dan push.
2. Di Vercel: **Add New → Project**, impor `meineruu/mdIR`, lalu atur:

| Pengaturan | Isi |
|---|---|
| Framework Preset | Vite |
| Root Directory | `app/frontend` |
| Build Command | `npm run build` (bawaan) |
| Output Directory | `dist` (bawaan) |
| Install Command | `npm install` (bawaan) |
| Environment Variables | tidak perlu |

3. Klik **Deploy**. Setiap push ke `main` akan men-deploy ulang otomatis.

Kalau halaman Vercel menampilkan "Backend belum siap", cek dua hal: apakah Space sudah menampilkan `{"status":"siap"}`, dan apakah `USERNAME` di `vercel.json` sudah diganti.
