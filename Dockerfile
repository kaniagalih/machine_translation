# Gunakan image resmi Python sebagai base image
FROM python:3.10-slim

# Tentukan direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt ke dalam container
COPY requirements.txt .

# Install dependencies yang ada di dalam requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi ke dalam container
COPY ./app/ /app/


# Tentukan port yang digunakan oleh aplikasi
EXPOSE 8501

# Jalankan aplikasi menggunakan Flask (atau bisa menggunakan Gunicorn untuk produksi)
CMD ["streamlit", "run", "dev/app.py"]