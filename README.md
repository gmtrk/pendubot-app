# Symulator Pendubota i Analiza Systemów Sterowania

## Wprowadzenie

Projekt ten jest częścią pracy magisterskiej pod tytułem **"PenduBot – Wirtualne stanowisko dydaktyczne do modelowania oraz ewaluacji algorytmów sterowania procesami nieliniowymi"**. Obejmuje implementację wirtualnego stanowiska dydaktycznego do symulacji, analizy i ewaluacji systemów sterowania dla nieliniowego układu Pendubota. Pendubot jest przykładem systemu niedoaktuowanego, gdzie liczba stopni swobody jest większa niż liczba dostępnych sygnałów sterujących, co czyni go interesującym obiektem badań w teorii sterowania.

Aplikacja umożliwia:
* Interaktywną konfigurację parametrów fizycznych Pendubota.
* Wizualizację zachowania systemu w czasie rzeczywistym.
* Wyświetlanie wykresów zmiennych stanu.
* Testowanie i porównywanie różnych strategii sterowania, w tym:
    * Regulatora PID (State-Feedback PID) z kompensacją grawitacji i interaktywnym strojeniem wzmocnień.
    * Regulatora Liniowo-Kwadratowego (LQR).
    * Agenta uczenia maszynowego ze wzmocnieniem opartego na algorytmie Proksymalnej Optymalizacji Polityki (PPO).

## 1. Konfiguracja Środowiska i Instalacja

### Wymagania Wstępne
* Python 3.8+
* `pip` (manager pakietów Pythona)
* `git` (do sklonowania repozytorium, opcjonalnie)

### Kroki Instalacyjne

1.  **Sklonuj repozytorium (jeśli dotyczy):**
    ```bash
    git clone https://github.com/gmtrk/pendubot-app.git
    cd pendubot-app
    ```
    Jeśli masz już pliki lokalnie, przejdź do głównego katalogu projektu (`pendubot-app`).

2.  **Utwórz i aktywuj wirtualne środowisko (zalecane):**
    ```bash
    python -m venv .venv
    ```
    Aktywacja:
    * Windows (Command Prompt):
        ```cmd
        .venv\Scripts\activate
        ```
    * Windows (PowerShell):
        ```powershell
        .venv\Scripts\Activate.ps1
        ```
        (Jeśli napotkasz problemy z polityką wykonywania skryptów, możesz potrzebować uruchomić: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)
    * Linux/macOS (Bash/Zsh):
        ```bash
        source .venv/bin/activate
        ```

3.  **Zainstaluj wymagane zależności:**

    ```bash
    pip install -r requirements.txt
    ```
    *Uwaga dotycząca Tkinter:* Tkinter jest zazwyczaj częścią standardowej biblioteki Pythona. Jednak w niektórych systemach Linux może być konieczne doinstalowanie go osobno, np.: `sudo apt-get update && sudo apt-get install python3-tk`.

## 2. Uruchamianie Aplikacji Symulacyjnej (GUI)

Po pomyślnej instalacji zależności, aplikację symulacyjną można uruchomić z głównego katalogu projektu:

```bash
python app/pendubot_app.py
```
Interfejs graficzny pozwoli na:
* Dostosowanie parametrów fizycznych Pendubota (masy, długości).
* Wybór jednej z predefiniowanych konfiguracji startowych/docelowych.
* Wybór metody sterowania: Brak, PID, LQR lub wytrenowany agent PPO.
* Interaktywne dostrajanie wzmocnień dla regulatora PID (Kp, Ki, Kd dla $q_1$ i $q_2$).
* Obserwację wizualizacji ruchu Pendubota oraz wykresów zmiennych stanu w czasie rzeczywistym.
* Monitorowanie aktualnie aplikowanego momentu sterującego $\tau_1$ oraz kumulacyjnych wskaźników jakości regulacji (IAE).

## 3. Trening Agenta PPO (Reinforcement Learning)

Aby móc korzystać z kontrolera "PPO RL Agent" w głównej aplikacji, należy użyć wytrenowanego już modelu. Możesz to zrobić, używając dołączonego modelu, albo wytrenować model samodzielnie, postępując zgodnie z poniższymi krokami.

1.  **Uruchamiaj z głównego katalogu projektu:**
    ```bash
    # Jeśli jesteś w głównym katalogu projektu (pendubot-app):
    python rl_training/train_ppo_pendubot.py
    ```

2.  **Proces Treningu:**
    * Skrypt `train_ppo_pendubot.py` wykorzystuje zdefiniowane środowisko `PendubotEnv` (z pliku `rl_training/pendubot_env.py`) oraz algorytm PPO z biblioteki Stable Baselines3.
    * Parametry fizyczne Pendubota podczas treningu są **stałe** (zgodnie z ostatnimi ustaleniami: $m_1=0.8, l_1=1.0, m_2=0.2, l_2=0.5$).
    * Konfiguracje docelowe są losowane na początku każdego epizodu (z wyłączeniem pozycji $(0,0)$).
    * Domyślne hiperparametry treningu (np. `TOTAL_TIMESTEPS = 10000000`, `N_ENVS = 16`) są zdefiniowane w skrypcie i mogą być modyfikowane.
    * Trening odbywa się na CPU (zgodnie z zaleceniami dla PPO z polityką MLP).

3.  **Wyniki Treningu:**
    * Wytrenowany model zostanie zapisany jako `ppo_pendubot_model.zip` w katalogu `data/`.
    * Punkty kontrolne modelu będą zapisywane w podkatalogu `data/ppo_pendubot_logs/`.
    * Logi do wizualizacji w TensorBoard będą zapisywane w `data/ppo_pendubot_tensorboard/`.

4.  **Monitorowanie Treningu (TensorBoard):**
    Podczas treningu (lub po jego zakończeniu) można uruchomić TensorBoard, aby śledzić postępy:
    ```bash
    # Z głównego katalogu projektu
    tensorboard --logdir data/ppo_pendubot_tensorboard/
    ```
    Otwórz wskazany adres (zazwyczaj `http://localhost:6006/`) w przeglądarce.

## 4. Replikacja Eksperymentów

Skrypt `run_controller_comparison.py` pozwala na zautomatyzowane przeprowadzenie serii symulacji w celu porównania wydajności zaimplementowanych kontrolerów (PID, LQR, PPO) dla różnych konfiguracji docelowych.

1.  **Upewnij się, że model PPO jest wytrenowany:** Skrypt będzie próbował załadować `ppo_pendubot_model.zip`.
2.  **Uruchom skrypt:**
    ```bash
    # Jeśli jesteś w głównym katalogu projektu (pendubot-app):
    python experiments/run_controller_comparison.py
    ```
3.  **Proces Eksperymentu:**
    * Skrypt wykorzystuje klasę `HeadlessPendubotSimulator` do przeprowadzania symulacji bez interfejsu graficznego.
    * Parametry fizyczne są stałe ($m_1=0.8, l_1=1.0, m_2=0.2, l_2=0.5$).
    * Każda symulacja trwa 10 sekund czasu symulacyjnego.
    * Testowane są wszystkie 4 niestabilne konfiguracje docelowe.
    * Dla regulatora PID stosowane są specyficzne, predefiniowane w skrypcie wzmocnienia $K_p$ dla każdego celu, a pozostałe wzmocnienia ($K_i, K_d$) brane są z wartości domyślnych.
    * Każdy scenariusz (cel + kontroler) jest powtarzany domyślnie 3 razy w celu uśrednienia wyników.
    * Zbierane są metryki: IAE dla $q_1$ i $q_2$, sumaryczny wysiłek sterujący oraz status ukończenia.

4.  **Wyniki Eksperymentu:**
    * Podsumowanie wyników zostanie wyświetlone w konsoli.
    * Szczegółowe wyniki zostaną zapisane do pliku `data/pendubot_controller_comparison_results.csv`.
