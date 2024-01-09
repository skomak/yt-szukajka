# yt-szukajka
Skrypt służy do rozmawiania z materiałami wideo / tekstem z użyciem sztucznej inteligencji - modelu whisper do transkrypcji i trurl-2 do analiz. Obsługuje język angielski i **polski**. 
Na tą chwilę działa jedynie dla materiałów rzędu kilku minut, ponieważ kontekst użytego modelu nie jest duży, a także ze względu na duże zapotrzebowanie pamięci GPU.

Skrypt działa domyślnie na GPU.
Testowane na GPU - NVIDIA RTX 4070 (12GB). Przykładowe zużycie GPU VRAM przy materiale 3-minutowym z promptem użytkownika jest na poziomie ~10GB.

Możliwe jest przestawienie na CPU i RAM przy zmianie na model w wersji bez kwantyzacji np. Voicelab/trurl-2-7b.

# instalacja
`$ source setup.sh`

# użycie
```bash
$ source set_env.sh
$ python yt_szukajka.py
```
# demo
![](demo.gif)
