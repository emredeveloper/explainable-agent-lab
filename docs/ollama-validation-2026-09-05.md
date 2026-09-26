# Ollama ile canlı doğrulama — 5 Eylül 2026

Yerel `mem-llm-lfm2.5:2.6b-q8` modeliyle araç çağrıları, hata sonrası toparlanma,
streaming ve sıralı çok ajanlı görevler çalıştırıldı. Son canlı koşuda altı
senaryonun tanımlı kontrolleri geçti. Bu sonuç, modelin her cümlesinin doğru
olduğu veya tüm modellerin desteklendiği anlamına gelmez.

## Yeniden üretilen sorunlar ve değişiklikler

| Sorun | Önceki gözlem | Değişiklik / doğrulama |
|---|---|---|
| Metin biçimindeki araç çağrısı final cevap sayılıyor | Dosya okunduktan sonra `<\|tool_call_start\|>[calculate_math(expression='37*12')]<\|tool_call_end\|>` kullanıcıya dönüyordu | Tek araç ve tek literal argüman AST ile çözümleniyor; Python kodu çalıştırılmıyor. Akış artık dosya → hesaplama → cevap |
| Görev bittiği halde ilgisiz araç seçiliyor | `444.0` hesaplandıktan sonra dosya listeleme çağrısı üretildi | Başarılı araç sonucunun ardından asıl görev tamamlandıysa final cevap verilmesi açıkça isteniyor |
| Streaming token sayısı sıfır | Doğru `444.0` cevabında kullanım `0` | Usage chunk isteniyor. Son canlı streaming koşusunda toplam `1373` token kaydedildi |
| Araç istisnası koşuyu kesebiliyor | Özel araç istisnaları doğrudan dışarı çıkıyordu | İstisna `ERROR:` sonucuna çevriliyor. Gerçek LLM ilk `TimeoutError` sonrasında tekrar deneyip `AMBER-7391` buldu |
| Başarı sayacı modelin hata analizine dayanıyor | Tekrar başarısız olan bir girişim iyileşme sayılabiliyordu | Deneme ve gözlenen başarılı araç tekrarı ayrı sayılıyor; görev tamamlanması iddia edilmiyor |
| Bağımlı ajan önceki sonucu almıyor | Yalnızca önceden yazılmış görev metni aktarılıyordu | Önceki sonuçlar ayrı `context` ile aktarılıyor. Aracı olmayan ikinci ajan ilk ajanın kodunu doğru döndürdü |
| Boş araç kayıt sistemi global araçlara dönüyor | Boş sözlük `or AVAILABLE_TOOLS` nedeniyle tüm araçları açabiliyordu | `None` ile boş kayıt ayrıldı; sezgisel yönlendirme de kapalı araçları eklemiyor |
| Faithfulness yanlış olumlu sonuç veriyor | Araç `848.0`, cevap `999.0` olsa da alternatif cevaptan farklılık olumlu sayılabiliyordu | Alternatif cevaptan farklılık olumlu kanıt olmaktan çıkarıldı; hata çıktıları desteğe katılmıyor. Regresyon testi yanlış cevabı reddediyor |
| Yardımcı cevap çağrıları toplam tüketimde yok | Yalnızca adımların tokenları toplanıyordu | `RunTrace.total_usage` yardımcı çağrıları da içeriyor; iki ayrı koşuda sayaçların karışmadığı test edildi |

## Son canlı koşu

Kaynak: [summary.json](../runs/live-checks/a2587456/summary.json).
Bu yerel izler Git tarafından yok sayılan `runs/` klasöründedir.

| Senaryo | Sonuç | Sınanan koşul |
|---|---|---|
| Aritmetik | `848.0` | Beklenen cevap, ham protokol sızıntısı yok |
| Dosya → hesaplama | `444.0` | Her iki araç çalıştı; aşağıdaki para birimi uyarısı üretildi |
| Streaming | `444.0`, 1373 token | Cevap ve kullanım sayacı |
| Geçici araç hatası | `AMBER-7391` | 1 deneme, 1 başarılı araç tekrarı |
| Kalıcı araç hatası | Açık başarısızlık açıklaması | Başarılı tekrar yok, yanlış olumlu faithfulness yok |
| Bağımlı ekip | İkinci ajan `AMBER-7391` döndürdü | Planlama, yürütme ve sentez gerçek LLM ile |

Ara koşudaki kalıcı hata fikstüründe alternatif çalışan bir araç da açık
bırakılmıştı. Model o aracı kullanarak gerçekten toparlandı; bu nedenle fikstürün
“toparlanma olmamalı” beklentisi yanlıştı. Son koşuda yalnızca bozuk araç açılarak
kalıcı hata senaryosu izole edildi. Önceki kayıtlar silinmedi.

## Kalan sınırlar

- Model, para birimi belirtilmeyen envanter cevabına `$` ekledi. Sayısal sonuç
  doğru olsa da bu ek bilgi doğrulanmış değildir. Cevap değiştirilmeden korunur;
  sistem uyarı verir ve `likely_faithful=False` işaretler. Canlı kontrol bu
  uyarının üretildiğini doğrular, modelin ek bilgi uydurmadığını iddia etmez.
- Faithfulness sözcük/sayı örtüşmesi sezgisidir. Anlamsal doğruluk, çelişki veya
  nedensel açıklanabilirlik garantisi değildir. Para birimi kontrolü de yalnızca
  açık semboller için dar bir kontrol sağlar.
- Etiketli protokolde birden fazla çağrı, birden fazla argüman veya çalıştırılabilir
  ifade desteklenmez; açık hata döner. Genel yerel model protokol desteği değildir.
- Token toplamı sağlayıcının bildirdiği kullanımı ölçer. Kullanım döndürmeyen
  başarısız istekler ölçülemez. Eşzamanlı ajanlarda istemci paylaşılmamalıdır.
- Canlı test tek kurulu modelle yapıldı. Harici arama, diğer sağlayıcıların native
  tool API'leri ve geniş benchmark veri kümeleri bu doğrulamanın kapsamında değil.

## Kontroller

- Python 3.14 yerel sanal ortam: **78 pytest testi geçti**.
- Ruff lint ve biçim kontrolü: **geçti**, 37 Python dosyası.
- Kaynak derleme kontrolü: **geçti**.
- Wheel ve sdist oluşturma: **geçti**.
- Twine metadata kontrolü: **iki paket de geçti**.
- Eksik yerel build araçları için yalnızca proje `.venv` ortamına `setuptools`
  ve `wheel` yüklendi. Paket yayınlanmadı; sürüm numarası değiştirilmedi.

Tekrarlamak için depo kökünde:

```powershell
.\.venv\Scripts\python.exe scripts/live_ollama_check.py --model mem-llm-lfm2.5:2.6b-q8 --repeat 2
.\.venv\Scripts\python.exe -m pytest -q
```

Her canlı çalıştırma yeni bir iz klasörü oluşturur. `summary.json` tanımlı
kontrolleri, uyarıları ve tek tek ham izlerin yollarını içerir.
