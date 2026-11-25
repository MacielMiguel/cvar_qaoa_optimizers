# Log das alterações — cvar_qaoa_optimizers_v2

Data: 24/11/2026. Base: commit `092332f` da pasta `cvar_qaoa_optimizers` (o original não foi modificado).
Os IDs (A1, B3, …) são os do documento "cvar_qaoa_optimizers: análise e plano de mudanças".

**Nada foi commitado.** As alterações estão na árvore de trabalho da v2, então
`git status` e `git diff` dentro de `cvar_qaoa_optimizers_v2` mostram tudo em relação ao commit original.

---

## 1. Como verificar rapidamente

```bash
cd cvar_qaoa_optimizers_v2
pip install -e .[pdfo,dev]
pytest                                     # 30 testes, todos passam (~5 s)
git diff -- src suites tests               # todas as mudanças de código
python main.py paper --quick               # roda as 3 suítes em versão mínima (~1 min)
python scripts/replot_paper.py results/paper   # regenera as figuras do artigo a partir dos JSON
python main.py paper --workers 4           # refaz todos os experimentos do artigo (~15 min com 2 núcleos)
```

---

## 2. Correções de código (o que mudou e por quê)

### `src/quantum_core/solver.py` — A1 (ordem dos bits)
- **Antes:** `brute_force` lia o estado com `format(state, '0nb')`, que coloca o qubit n−1 no caractere 0. O Qiskit usa o qubit 0 como bit menos significativo. Em grafos não simétricos, os "estados ótimos" usados para a probabilidade de sucesso eram os de outro grafo (18 de 20 instâncias erradas na Suite 3 antiga).
- **Agora:** nova função `state_to_bitstring(state, n)` (inverte a string), usada em todo lugar que converte inteiro → bitstring. A convenção está documentada no topo do arquivo.
- **Teste novo:** `test_brute_force_matches_quantum_operator` compara o brute force com o mínimo do operador quântico em 10 grafos aleatórios.

### `src/problems/maxcut/hamiltonian.py` — A3 (escala do Hamiltoniano)
- **Antes:** o operador quântico era H = Σ Z_iZ_j (energia = |E| − 2·corte, ótimo −8 no grafo de 6 nós), o `evaluate` devolvia −corte (ótimo −9) e a docstring/artigo diziam H_C = −½Σ(I − Z_iZ_j).
- **Agora:** o operador é exatamente H_C = ½Σ Z_iZ_j − |E|/2·I, cuja energia é −corte em todos os estados. Código, docstring e Eq. (1) do artigo agora coincidem. Arestas duplicadas (i,j)/(j,i) são contadas uma vez. Removida a dependência de `rustworkx`. Lista de arestas vazia sem `num_qubits` agora dá erro claro.
- **Teste novo:** `test_operator_energy_equals_minus_cut_for_every_state`.
- **Consequência:** os eixos das figuras mudaram de escala (ótimo agora é −9, não −8).

### `src/problems/maxcut/solvers.py` — A1
- `BruteForceSolver` usa `state_to_bitstring`.

### `src/quantum_core/experiment.py` — A4, A5, desempenho
- **A4 (reprodutibilidade):** o ponto inicial usa `np.random.default_rng(seed)` (antes: `np.random.seed(42)` global) e o `StatevectorSampler` recebe um *Generator* com seed (antes: sem seed). Com um inteiro, o Qiskit repetiria o mesmo ruído em toda avaliação; com um Generator, o ruído é novo a cada avaliação mas reprodutível.
- **A5 (orçamento):** novo parâmetro `max_evals` = orçamento em **avaliações do circuito**, igual para todos os otimizadores. Só as primeiras `max_evals` avaliações são executadas e registradas; se o otimizador pedir mais, recebe o último custo sem rodar o circuito. `maxiter` continua aceito como sinônimo (suítes MaxMulticut).
- **Resultado:** agora é um dict com o melhor ponto registrado, `nfev` (avaliações usadas) e `nfev_requested` (pedidas pelo otimizador). O resultado inclui `seed`, `max_evals`, `num_params` e a instância (arestas).
- **Desempenho:** o ansatz é decomposto uma única vez em portas elementares antes do loop. Mesmo resultado numérico, cerca de 40× mais rápido no QAOA (0,6 s em vez de 25,8 s para 26 avaliações com 8 qubits, p = 2).
- **Novo `run_many(tasks, workers)`:** roda execuções em paralelo (processos). A ordem dos resultados é preservada e o paralelismo não muda os números (testado).

### `src/quantum_core/optimizers.py` — A5
- Nelder-Mead e Powell recebem também `maxfev` (limite de avaliações, não só de iterações).
- Simulated Annealing (`dual_annealing`) recebe `maxfun` = orçamento, e os limites padrão passaram de [−π, π] para [0, 2π], o mesmo intervalo do ponto inicial.

### `src/quantum_core/display.py` — B3, B5, B6, C3, B2
- Tudo em inglês; removidos os banners de parâmetros dentro das figuras (vão na legenda do artigo e no `config.json`).
- Curvas com mediana + faixa interquartil sobre as seeds; eixo x = avaliações do circuito.
- **B3:** `cm.get_cmap` (removido no matplotlib 3.9) → `matplotlib.colormaps[...]`. Antes, a Suite 3 quebrava numa instalação nova.
- **B5:** legenda do histograma com `Patch` (antes todas as entradas apareciam azuis).
- **B6:** removido o corte silencioso em 50 iterações do benchmark.
- **B2:** as funções aceitam dict ou lista e agrupam pela chave certa; nova `plot_curves_by` para agrupar por qualquer campo.
- Tamanhos pensados para uma coluna do artigo; legendas fora das curvas.
- O benchmark de otimizadores mostra o "melhor custo até agora" (padrão em benchmarks).

### `src/quantum_core/results_io.py` — B4
- **Antes:** o campo `optimizer_result` saía sempre `{}` (testava `hasattr(r, 'x')` num dict).
- **Agora:** salva o resultado do otimizador e, por execução, `shots`, `seed`, `max_evals`, `num_params` e a instância. Cada suíte grava também `suite{N}_config.json` com a configuração e as versões das bibliotecas. No CSV, os estados ótimos são escritos na ordem dos vértices.

### `src/quantum_core/cvar_cost.py` — B9
- Só a docstring (`optimal_state` → `optimal_states`).

### Suítes MaxCut — C2 (configurações do artigo)
- `suite1.py`: grafo fixo de 6 nós, p = 1, 7 valores de α, COBYLA, 100 avaliações, 10 seeds; gera `suite1_alpha_sweep.png` e `suite1_distribution.png` (VQE, α = 0,25, seed 0).
- `suite2.py`: 11 otimizadores (6 SciPy + 5 PDFO), **todos com α = 0,5** e 200 avaliações, 10 seeds; gera `suite2_scipy.png` e `suite2_pdfo.png`.
  *Decisão minha:* antes, a figura SciPy usava α = 1 e a PDFO α = 0,5. Unifiquei em 0,5 para que todos os otimizadores sejam comparáveis. Se preferir manter α diferentes, é só mudar `alpha` no config.
- `suite3.py`: 60 grafos aleatórios de 8 nós (seeds 800 a 859), densidade 0,5, p ∈ {1, 2}, α ∈ {0,01; 0,25; 0,5; 1}, COBYLA, 100 avaliações.
  *Decisão minha:* usei 60 instâncias (o código tinha 20; a figura do relatório indicava 60).

### Suítes MaxMulticut — B1, B2 (ajustes mínimos para continuarem funcionando)
- Suite 3: `random(...)` → `random_instance(...)` (antes: `TypeError`).
- Suites 2 e 3 usam a função de gráfico adequada (antes desenhavam uma única curva). Todas salvam `config.json`.
- **Não corrigido (A2):** com 3 cores, o operador quântico continua sem penalidade para códigos de cor inválidos. Documentado no README como limitação; não afeta o artigo.

### `main.py`, `run.sh` — D2, C7
- `python main.py <suítes> [--workers N] [--quick] [--verbose]`, ou `python main.py paper` para as três suítes do artigo. Logs do Qiskit silenciados.
- `run.sh` não tem mais o caminho fixo da máquina do Hugo.

### `scripts/replot_paper.py` (novo)
- Regenera as figuras do artigo a partir de `results/paper/*.json`, sem rodar as suítes. O histograma refaz uma única execução (determinística).

### Testes — D1
- `test_brute_force.py`: reescrito com asserções (antes só imprimia) e testes de regressão para A1 e A3.
- `test_experiment.py` (novo): reprodutibilidade por seed, respeito ao orçamento para 5 otimizadores, paralelo = serial, gravação de JSON/config e geração das 4 figuras.
- `test_solvers.py`: o teste do Simulated Annealing foi ajustado porque `maxiter` agora significa orçamento de avaliações.
- Resultado: **30 testes passam**.

### `README.md`, `pyproject.toml`, `.gitignore` — C1, C5, C6
- README: novas seções "Run Experiments", "Reproducing the Paper" (tabela figura → suíte → arquivo), "Suite Descriptions" e "Reproducibility" escritas a partir do código; licença GPL-3.0 (o arquivo `LICENSE` já era GPL-3.0; o README dizia "TBD").
- `pyproject.toml`: `pdfo` como extra opcional (`pip install -e .[pdfo]`); removidos `networkx`, `qiskit-algorithms` e `rustworkx` (não usados); `matplotlib>=3.9`; Hugo como autor; licença declarada.
- `.gitignore`: as execuções continuam ignoradas, exceto `results/paper/`, que fica versionada.

---

## 3. Experimentos executados

- Ambiente: Python 3.13.18, Qiskit 2.6.2, NumPy 2.5.3, SciPy 1.18.1, matplotlib 3.11.2, PDFO 2.2.0 (gravado em cada `config.json`).
- Comando: `python main.py paper --workers 2`. Total de 1320 execuções (Suite 1: 140, Suite 2: 220, Suite 3: 960), cerca de 15 min.
- Resultados brutos, CSV, configs e figuras: `results/paper/`.

### Principais números (os mesmos usados no texto do artigo)

**Suite 1** (grafo de 6 nós, ótimo −9, mediana de 10 seeds)

| | α = 0,01 | 0,05 | 0,1 | 0,25 | 0,5 | 0,75 | 1 |
|---|---|---|---|---|---|---|---|
| QAOA: CVaR final | −9 | −9 | −9 | −9 | −7,57 | −6,56 | −6,33 |
| QAOA: p_succ final | 0,08 | 0,13 | 0,17 | 0,26 | 0,26 | 0,21 | 0,22 |
| VQE: CVaR final | −9 | −9 | −9 | −9 | −9 | −8,99 | −8,33 |
| VQE: p_succ final | 0,04 | 0,08 | 0,12 | 0,30 | 0,53 | 0,76 | 0,75 |

Leitura nova: no VQE, a probabilidade de sucesso final acompanha α. O CVaR_α já é mínimo assim que o ótimo tem probabilidade α, então o otimizador não tem incentivo para concentrar mais.

**Suite 2** (α = 0,5, 200 avaliações, VQE, 10 seeds)

| Otimizador | Execuções que atingem −9 | Avaliações até −9 (mediana) |
|---|---|---|
| COBYLA (SciPy) | 8/10 | 37 |
| COBYLA (PDFO) | 9/10 | 42 |
| NEWUOA | 10/10 | 73 |
| LINCOA | 10/10 | 75 |
| Powell | 5/10 | 122 |
| BOBYQA | 3/10 | 53 |
| UOBYQA | 1/10 | — (o modelo inicial exige 325 avaliações) |
| Nelder-Mead | 1/10 | — |
| Dual annealing, BFGS, CG | 0/10 | — |

No QAOA, todos ficam entre −7,25 e −7,82.

**Suite 3** (60 grafos de 8 nós): fração de instâncias que atinge p_succ ≥ 10%

| | α = 0,01 | 0,25 | 0,5 | 1 |
|---|---|---|---|---|
| QAOA, p = 1 | 0,27 | 0,52 | 0,53 | 0,45 |
| QAOA, p = 2 | 0,27 | 0,70 | 0,63 | 0,60 |
| VQE, p = 1 | 0,12 | 1,00 | 0,92 | 0,73 |
| VQE, p = 2 | 0,10 | 0,88 | 0,73 | 0,65 |

O limiar de 1% é atingido em 95–100% das instâncias por todas as configurações.

**Mudança de conclusão:** com o bug de bits corrigido, a Suite 3 **confirma** Barkoutsos et al. (α intermediário > α = 1) e mostra que α muito pequeno é prejudicial. A versão anterior do artigo dizia "neither confirm nor contradict".

---

## 4. Alterações no artigo (`paper/`)

- **Abstract:** frases de resultado reescritas com as novas conclusões (Suite 3 confirma o artigo original; benchmark com orçamento igual).
- **Introdução:** a contribuição (iii) agora menciona o orçamento igual de avaliações.
- **Eq. (1):** mantida (agora coincide com o código) e acrescentada a frase "eigenvalue = minus the cut size".
- **Métodos:** parágrafo sobre orçamento em avaliações e seeds; setup com 10 seeds, α = 0,5 na Suite 2, mediana e IQR.
- **Tabela I:** tamanhos com seeds, α da Suite 2, nova linha "Budget".
- **Resultados IV-A, IV-B, IV-C:** reescritos a partir dos novos dados (números das tabelas acima). Ótimo agora −9.
- **Figuras 2 a 5:** substituídas pelas novas (inglês, sem banners, mediana e IQR). Legendas das figuras reescritas.
- **Discussão:** reescrita (CVaR confirma o artigo original; α define o p_succ alvo; o orçamento decide o ranking dos otimizadores). Limitações atualizadas: removidas as que foram resolvidas (seed única, orçamento desigual) e mantidas as outras.
- **Conclusão:** reescrita; nova seção **Code Availability** com o link do GitHub.
- Continua com **4 páginas**, compilado sem erros nem avisos.

---

## 5. O que você precisa conferir ou decidir

1. **Texto do artigo:** reescrevi as seções de resultados, discussão, conclusão e partes do abstract. Leiam com atenção, porque as conclusões mudaram. No AI disclaimer, isso precisa ser declarado (vai além de revisão).
2. **α = 0,5 em toda a Suite 2 e 60 instâncias na Suite 3** (decisões minhas, ver seção 2).
3. **Code Availability:** o link aponta para o GitHub, mas o GitHub ainda tem o código antigo. Para o link valer, é preciso fazer commit e push da v2 (posso preparar a mensagem de commit). Eu não fiz commit nem push.
4. **Author Contributions e Use of Generative AI:** continuam com `\ldots` para vocês preencherem.
5. **Não feito (não afeta o artigo):** A2 (penalidade de cores inválidas no MaxMulticut), A6 nas suítes MaxMulticut, B7–B8, C4, C8, D3–D6.
6. A pasta `cvar-quantum-main` ao lado é uma versão anterior do mesmo código (antes das últimas edições); não foi usada nem alterada.
