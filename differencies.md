# Differenze tra `Lecture2.ipynb` e `my_cnn.ipynb`

Entrambi i notebook affrontano lo stesso problema: **addestrare una rete neurale convoluzionale (CNN) per ricostruire immagini CT degradate** (sfocatura da movimento + rumore gaussiano) a partire dal dataset Mayo. Tuttavia, differiscono significativamente in struttura, robustezza e completezza.

---

## 1. Struttura e organizzazione

| Aspetto | `Lecture2.ipynb` | `my_cnn.ipynb` |
|---|---|---|
| **Numero di blocchi di codice** | 4 | 6 |
| **Scopo** | Appunti presi durante la lezione | Versione rielaborata e migliorata |
| **Ordine dei blocchi** | Dataset → CNN → Training → Visualizzazione | CNN → Setup completo → Training + salvataggio → Test → Analisi attivazioni → Conteggio parametri |

`Lecture2` segue l'ordine in cui il materiale è stato presentato in aula: prima il dataset, poi la rete, poi il training. `my_cnn` riorganizza il contenuto in modo più logico: prima la definizione della rete (indipendente dai dati), poi tutto il setup dell'ambiente, e infine una pipeline completa di training, validazione e analisi.

---

## 2. Gestione dei percorsi e dell'ambiente

| Aspetto | `Lecture2.ipynb` | `my_cnn.ipynb` |
|---|---|---|
| **Percorso dati** | Hardcoded (`"Mayo/train"`) | Ricerca automatica della cartella `Mayo` risalendo l'albero delle directory |
| **Import di IPPy** | Diretto (`from IPPy import operators`) | Dinamico tramite `importlib.util` |
| **Cartella pesi** | Non gestita | Creata automaticamente (`weights_dir.mkdir(exist_ok=True)`) |
| **Portabilità** | Funziona solo se il notebook viene eseguito dalla directory corretta | Funziona indipendentemente dalla directory di esecuzione |

`my_cnn` è significativamente più robusto: non dipende dalla directory di lavoro corrente e gestisce automaticamente la creazione delle cartelle necessarie.

---

## 3. Selezione del dispositivo di calcolo

| Aspetto | `Lecture2.ipynb` | `my_cnn.ipynb` |
|---|---|---|
| **Device** | Hardcoded a `"mps"` | Rilevamento automatico (`cuda` → `mps` → `cpu`) tramite funzione `get_device()` |

In `Lecture2`, il dispositivo è fissato a `"mps"` (Apple Silicon), il che significa che il notebook **non funziona** su macchine senza GPU Apple. `my_cnn` rileva automaticamente il dispositivo disponibile, rendendo il codice compatibile con qualsiasi piattaforma.

---

## 4. Generazione del rumore

| Aspetto | `Lecture2.ipynb` | `my_cnn.ipynb` |
|---|---|---|
| **Funzione di rumore** | `utilities.gaussian_noise()` dalla libreria IPPy | Funzione locale `gaussian_noise()` definita nel notebook stesso |

`my_cnn` definisce la funzione di rumore direttamente nel notebook, rendendolo autocontenuto e indipendente da dettagli implementativi della libreria del corso.

---

## 5. Parametri del training

| Parametro | `Lecture2.ipynb` | `my_cnn.ipynb` |
|---|---|---|
| **Numero di epoche** | 3 | 30 |
| **Batch size** | 4 | 8 |
| **Numero di filtri** | 32 | 32 |
| **Learning rate** | 1e-3 | 1e-3 |
| **Scheduler** | `CosineAnnealingLR` (T_max=2) | Nessuno |
| **Dataset di test separato** | No | Sì |
| **Barra di progresso** | No (solo `print` con `\r`) | Sì (`tqdm`) |

Differenza fondamentale: `Lecture2` usa sole 3 epoche (dimostrazione veloce in aula), mentre `my_cnn` ne usa 30 per un addestramento più completo. `Lecture2` include uno scheduler del learning rate (`CosineAnnealingLR`) che `my_cnn` non usa.

---

## 6. Operatore di degradazione (Blurring)

| Parametro | `Lecture2.ipynb` | `my_cnn.ipynb` |
|---|---|---|
| **Tipo kernel** | `"motion"` | `"motion"` |
| **Dimensione kernel** | Non specificata esplicitamente | `kernel_size=9` |
| **Angolo** | `motion_angle=45` | `motion_angle=20` |
| **Varianza** | `kernel_variance=1.0` | Non specificata |

I due notebook usano parametri diversi per l'operatore di sfocatura, il che significa che risolvono problemi inversi leggermente differenti.

---

## 7. Salvataggio e ricaricamento del modello

| Aspetto | `Lecture2.ipynb` | `my_cnn.ipynb` |
|---|---|---|
| **Salvataggio pesi** | Non presente | `torch.save(model.state_dict(), weights_path)` |
| **Ricaricamento pesi** | Non presente | Sì, con verifica su immagine di test |
| **Verifica round-trip** | No | Sì |

`my_cnn` include una fase completa di salvataggio su disco e ricaricamento dei pesi, verificando che il modello ricaricato produca risultati validi. Questo è un pattern fondamentale per applicazioni reali.

---

## 8. Analisi e interpretabilità del modello

| Aspetto | `Lecture2.ipynb` | `my_cnn.ipynb` |
|---|---|---|
| **Visualizzazione feature maps** | Non presente | Forward hooks su `conv1` con griglia 4×4 |
| **Conteggio parametri** | Non presente | Presente |
| **Confronto visivo** | Solo immagine originale | Ground truth, misura degradata, ricostruzione + curva di loss |

`my_cnn` offre strumenti di analisi assenti in `Lecture2`:
- **Forward hooks** per visualizzare le attivazioni intermedie della rete (utile per capire cosa la CNN ha imparato).
- **Conteggio dei parametri** addestrabili.
- **Grafico della training loss** per monitorare la convergenza.

---

## 9. Ordine delle operazioni nel training loop

| Aspetto | `Lecture2.ipynb` | `my_cnn.ipynb` |
|---|---|---|
| **Ordine** | `loss.backward()` → `optimizer.step()` → `scheduler.step()` → `optimizer.zero_grad()` | `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` |

`Lecture2` chiama `optimizer.zero_grad()` **dopo** `optimizer.step()` (alla fine dell'iterazione), mentre `my_cnn` segue il pattern più convenzionale di azzerare i gradienti **prima** del backward pass (tramite la sequenza implicita). In pratica il risultato è equivalente, ma l'ordine di `my_cnn` è più leggibile e conforme alle best practice.

---

## 10. Riepilogo

`Lecture2.ipynb` rappresenta degli **appunti presi in aula** durante la lezione, con codice funzionante ma minimale e privo di robustezza (percorsi hardcoded, poche epoche, nessun salvataggio).

`my_cnn.ipynb` è una **rielaborazione personale** che migliora significativamente il codice sotto diversi aspetti:
- **Portabilità**: ricerca automatica dei percorsi e rilevamento del dispositivo.
- **Completezza**: training più lungo, dataset di test separato, salvataggio/caricamento dei pesi.
- **Interpretabilità**: visualizzazione delle feature maps, conteggio parametri, curva di loss.
- **Usabilità**: barra di progresso `tqdm`, output informativi durante il training.
