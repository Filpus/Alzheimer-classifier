# Konfiguracja DVC dla zespołu

Niniejsza instrukcja opisuje proces podłączenia lokalnego środowiska programistycznego do zdalnego magazynu danych DVC na platformie DagsHub. Dzięki temu można pobrać surowy zbiór EEG oraz synchronizować kolejne wyniki przetwarzania, takie jak tensory i osadzenia.

## Wymagania wstępne

- Zainstalowany i skonfigurowany system **Git**.
- Dostęp do repozytorium projektu.
- Nadane uprawnienia współpracownika w projekcie na DagsHub.

## 1. Pobranie projektu i przygotowanie środowiska

Sklonuj repozytorium i przejdź do katalogu projektu:

```bash
git clone https://github.com/Filpus/Alzheimer-classifier.git
cd Alzheimer-classifier
```

Jeśli korzystasz z własnego środowiska Pythona, aktywuj je przed wykonaniem kolejnych poleceń.

## 2. Autoryzacja i pobranie tokenu DagsHub

Ponieważ logowanie do DagsHub odbywa się poprzez protokół OAuth GitHub, klasyczne hasło nie zadziała w konfiguracji DVC. Wymagane jest użycie tokenu dostępu.

1. Zaloguj się na [dagshub.com](https://dagshub.com).
2. Wejdź w **Settings** klikając ikonę profilu w prawym górnym rogu.
3. Wybierz zakładkę **Tokens** z menu po lewej stronie.
4. Skopiuj wartość oznaczoną jako **Default Token**.

## 3. Lokalna konfiguracja połączenia DVC

W terminalu wpisz poniższe polecenia, podstawiając swój login z DagsHub oraz skopiowany token:

```bash
dvc remote modify origin --local user TWÓJ_LOGIN_DAGSHUB
dvc remote modify origin --local password TWÓJ_SKOPIOWANY_TOKEN
```

Flaga `--local` jest krytyczna. Zapisuje poświadczenia w pliku `.dvc/config.local`, który jest ignorowany przez Git, więc token nie zostanie przypadkowo upubliczniony.

## 4. Pobieranie danych

Po skonfigurowaniu poświadczeń uruchom proces pobierania plików binarnych. Serwer DagsHub posiada limit jednoczesnych połączeń, dlatego wymuszamy ograniczenie do 4 wątków:

```bash
dvc pull 
```

Po zakończeniu operacji w strukturze projektu pojawią się fizyczne pliki `.set` oraz `.fdt` w lokalizacji `data/raw/eeg/` oraz powiązane metadane.

## 5. Protokół aktualizacji danych

Jeśli zadanie wymaga dodania nowych danych, na przykład wygenerowania przetworzonych tensorów do `data/processed/`, postępuj według poniższego schematu:

1. Umieść wygenerowane pliki w odpowiednim podfolderze w katalogu `data/`.
2. Dodaj folder do śledzenia w DVC:

	```bash
	dvc add data/processed
	```

3. Zatwierdź zmiany w konfiguracji DVC w systemie Git. DVC automatycznie zaktualizuje plik `.gitignore`:

	```bash
	git add data/processed.dvc data/.gitignore
	git commit -m "data: aktualizacja tensorów wejściowych"
	```

4. Wypchnij ciężkie pliki binarne do magazynu DagsHub:

	```bash
	dvc push 
	```

5. Wypchnij kod i wskaźniki konfiguracji do repozytorium GitHub:

	```bash
	git push
	```

## Uwaga końcowa

Zawsze trzymaj poświadczenia w pliku lokalnym `.dvc/config.local` i nie wpisuj tokenu bezpośrednio do plików śledzonych przez Git.

