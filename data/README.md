# Marta Rosas nª 20260651


# Dataset -- COMPAS Recidivism (ProPublica)

## The problem

In 2016, ProPublica investigated COMPAS, a risk-assessment algorithm
actually used by courts in Broward County, Florida, to help inform
bail and sentencing decisions. COMPAS scores a defendant's likelihood
of reoffending on a 1-10 scale; judges could see that score when
deciding, among other things, whether someone should be released
before trial. ProPublica obtained COMPAS's scores for thousands of
defendants and matched them against what actually happened over the
following two years, then published the data.

This dataset is that data: each row is one defendant, with their
demographics and criminal history at the time of screening, COMPAS's
own risk score for them, and whether they were actually rearrested
within two years.

**Your task:** predict `two_year_recid` -- will this person be
rearrested within two years? -- from the case facts. Once you have a
model, the more interesting question is the one ProPublica actually
asked: is it equally accurate for everyone, or does it get things
wrong more often, in a particular direction, for some groups than
others? `race` is deliberately excluded from the model's own inputs
(see `config.yaml` and `src/preprocessing.py`) so it can be used
afterward purely to check this, in `src/evaluate.py`.

Before any of that: look at the data first. It comes from a real
system with real data-entry and record-keeping quirks -- don't assume
every column is clean or consistent just because it loads without
error.

## Data dictionary

| column | type | description | notable values |
|--------|------|--------------|------------------|
| `id` | identifier | internal record id | not a model feature |
| `sex` | categorical | defendant's sex | `Male`, `Female` |
| `age` | numeric | defendant's age (years) at screening | |
| `age_cat` | categorical | age bucket | `Less than 25`, `25 - 45`, `Greater than 45` |
| `race` | categorical | defendant's race, as recorded | `African-American`, `Caucasian`, `Hispanic`, `Asian`, `Native American`, `Other`; excluded from model features, used only to audit fairness |
| `juv_fel_count` | numeric | number of prior juvenile felony offenses | |
| `juv_misd_count` | numeric | number of prior juvenile misdemeanor offenses | |
| `juv_other_count` | numeric | number of other prior juvenile offenses | |
| `juvenile_total` | numeric | total juvenile offenses | |
| `priors_count` | numeric | number of prior adult offenses | |
| `prior_offenses` | numeric | number of prior offenses | |
| `age_in_months` | numeric | age expressed in months | |
| `c_charge_degree` | categorical | degree of the current charge | `F` (felony), `M` (misdemeanor) |
| `decile_score` | numeric | COMPAS's own risk score | 1 (lowest risk) to 10 (highest risk); excluded from model features, used only for comparison |
| `score_text` | categorical | COMPAS's own risk category | `Low`, `Medium`, `High`; excluded from model features, used only for comparison |
| `two_year_recid` | binary | **target** -- was this person rearrested within two years? | `0` = no, `1` = yes |

Source: derived from [propublica/compas-analysis](https://github.com/propublica/compas-analysis) (the data behind the "Machine Bias" investigation). Personally-identifying columns (name, date of birth, case numbers, charge descriptions) were removed.




# progresso semana1 
Alterei o código de modo a comparar os modelos Logistic Regression e Decision Tree.
O Logistic Regression obteve uma accuracy de 0.679 no treino e 0.678 no teste, apresentando uma diferença muito reduzida entre ambos (0.001). Já a Decision Tree obteve 0.829 no treino, mas apenas 0.629 no teste, com um gap de 0.200.
Assim, com base nos resultados obtidos, o Logistic Regression apresenta, até ao agora, melhor capacidade de generalização, apesar de ter uma accuracy inferior no conjunto de treino.

# progresso semana 2
Esta semana, criei o ficheiro data_diagnostics.py, onde implementei três funções principais. A test_missingness_mechanism que usa o Chi-squared test e do carmers v. A flag_invalid_values e a find_duplicates (que permite identificar linhas duplicadas e, quando aplicável, IDs repetidos). Este ficheiro inclui também a função _cramers_v, usada dentro da função test_missingness_mechanism, para calcular a associação entre variáveis categóricas. Criei tambem pequenas explicaçoes sobre cada função ao pé do codigo para nao me ir perdendo.

No preprocessing.py, adicionei a função clean_dataset, esta utiliza a função _canonicalize_categories para uniformizar categorias que apareciam representadas de diferentes formas. Por exemplo, african-american, African American e AFRICAN-AMERICAN passam a ser representados de forma consistente como African-American (= ao ex da aula). Esta função usa tambem a flag_invalid_values que está no data_diagnostics.py e ptt importei para o preprocessing.py.

Adicionei ao config.yaml as regras de diagnóstico e limpeza dos dados, defini também regras para uniformizar categorias e indiquei as variáveis redundantes a remover, identificadas anteriormente através da análise de correlação e VIF. Adicionei também ao config.yaml a configuração do preprocessing.

No main.py, importei a função clean_dataset e alterei a função def main para que o df_clean fosse efetivamente utilizado no preprocessing.

Depois destas alterações, testei os modelos Logistic Regression e Decision Tree, usei test_size = 0.2 e random_state = 42 (tal como pre-definido). A Logistic Regression apresentou 0.678 de accuracy no treino e 0.655 no teste, com um gap de 0.023. O F1-score da classe 1 foi 0.57 e o macro F1-score 0.64. A Decision Tree apresentou 0.799 de accuracy no treino e 0.606 no teste, com um gap de 0.193. O F1-score da classe 1 foi 0.50 e o macro F1-score 0.59. A diferença maior entre treino e teste na Decision Tree sugere overfiting.

Na Logistic Regression, obtive relativamente ao FPR 0.30 para African-American, 0.16 para Caucasian, 0.19 para Hispanic e 0.10 para Other. Na Decision Tree, os valores foram 0.33, 0.19, 0.16 e 0.18, respetivamente. 
A comparação dos modelos permitiu analisar o seu desempenho através da accuracy e dos F1-scores, enquanto a análise da FPR por raça permitiu avaliar diferenças entre grupos.

# progresso semana 3

Na Semana 3, alterei o preprocessing.py adicionando as funções split_features_target() e build_preprocessor(). Também adicionei a função split_dev_test(), utilizada posteriormente para separar o development set do locked test set.
No model.py, foi adicionada a função build_model(), permitindo selecionar e configurar diferentes modelos (random forest, dummy, descison tree e logistic regression) através do config.yaml. Passei também a definir as configurações de preprocessing no config.yaml, incluindo o encoder, scaler, random_state, features numéricas e categóricas e estratégias de imputação.

Implementei já a parte do cross-validation relativa à Semana 4.
No evaluate.py, adicionei as funções cross_validate_pipeline(), cv_report(), oof_classification_report() e fairness_report(). No config.yaml, acrescentei as configurações da cross-validation e do locked test set, substituindo o antigo split direto de 80/20 entre treino e teste. O results.py foi também atualizado para guardar as configurações e resultados da cross-validation. Por fim, alterei o main.py para integrar este novo fluxo, utilizando o development set para a cross-validation e mantendo o test set separado para a avaliação final.

Na Semana 3, os modelos foram avaliados através de uma única divisão de 80/20 entre treino e teste. A Logistic Regression obteve a maior accuracy no conjunto de teste, com 0.673, seguida pelo Random Forest com 0.648 e pela Decision Tree com 0.625, enquanto o modelo Dummy obteve 0.546. A Logistic Regression também apresentou um train–test gap muito pequeno (+0.003), em comparação com +0.148 para o Random Forest e +0.171 para a Decision Tree o que sugere um maior risco de overfitting.

Na Semana 4, a avaliação foi melhorada através da criação do test set (1443 obs), que não foi utilizado durante a seleção dos modelos, e da aplicação de 5-fold cross-validation no conjunto de treino (5771 obs). 
Na cross-validation, a Logistic Regression obteve uma mean validation accuracy de 0.672, com std de 0.013 e um train–validation gap de +0.003. O Random Forest obteve uma mean validation accuracy de 0.650, com std de 0.018 e gap de +0.083, enquanto a Decision Tree obteve 0.609, std de 0.017 e gap de +0.085. O Dummy apresentou 0.549, com std de 0.000 e gap 0.

A mean validation accuracy da cross-validation é mais confiavel do que a accuracy de uma única divisão (Semana 3), pq a cross-validation avalia o modelo em cinco diferentes folds dando uma ideia mais consistente de como o modelo pode generalizar para novos dados e alem disso o TargetEncoder é ajustado apenas com os dados de treino de cada fold, evitando target leakage para o conjunto de validação.
A conclusão da Semana 3 mantém-se com o cross-validation, a Logistic Regression passou de 0.673 de accuracy no teste da Semana 3 para 0.672 std 0.013 na Semana 4, enquanto o Random Forest passou de 0.648 para 0.650 std 0.018 e a Decision Tree de 0.625 para 0.609 std 0.017. 
Além disso, o train–validation gap da Logistic Regression manteve-se muito pequeno (+0.003), exatamente igual ao train–test gap observado na Semana 3 (+0.003), mostrando um desempenho bastante estável. 

# progresso semana 5

(o hyperparameter_tuning ja estava implementado pois fiz na semana 4, em que nao houve avaliação semanal)

Esta semana alterei o models.py adicionando o addaboost(), o oblique_random_forest() e o dicionário: _MODELS. No config.yaml adicionei a secção 5.1 relativa ao tuning.search_spaces. 
 
Resultados:

Modelo	           Accuracy média	Desvio-padrão
Random Forest	         0,681	       0,015
Decision Tree	         0,672	       0,017
Oblique Random Forest	 0,679	       0,013
AdaBoost	             0,682	       0,017
Logistic Regression	     0,671	       0,015

O AdaBoost tem a maior accuracy média nesta baseline (0,682), mas a diferença para Random Forest (0,681) é de apenas 0,001, ou seja, não é suficiente, por si só, para concluir que um modelo é realmente superior.
A regressão logística apresenta 0,671, cerca de 1,1 pontos percentuais abaixo do AdaBoost. Continua a ser uma baseline importante para avaliar se as novas features melhoram um modelo linear.

Modelo	               Gap inicial	Gap após tuning
Random Forest	          0,082	         0,009
Decision Tree	          0,085	         0,013
Oblique Random Forest	  0,114	         0,013
AdaBoost	              0,006	         0,004
Logistic Regression	      0,003	         0,005

A Random Forest, a Decision Tree e a Oblique Random Forest mostram gaps iniciais relativamente elevados. Com o tunning reduz bastante esses gaps. Isto sugere que limitar a complexidade dos modelos ajuda a controlar o overfitting.
A regressão logística e o AdaBoost já apresentam gaps pequenos antes do tuning mas isso não significa que sejam necessariamente melhores.

Challenge:
adicionei add_engineered_features() ao preprocessing.py onde criei as variaveis "priors_per_adult_year","juvenile_offense_total" e "age_priors_interaction" e no main, chamei o add_engineered_features.

No config.yaml adicionei à lista as 3 features numericas que criei.

Resultados dos 3 modelos:

Modelo	              Sem as features  Com as features	  Diferença 
Random Forest	           0,681	     0,674	       −0,007
AdaBoost	               0,682	     0,677	       −0,005
Logistic Regression	       0,671	     0,671	        0,000

A Random Forest e o AdaBoost apresentam pequenas diminuições, enquanto a regressão logística mantém a mesma média, após arredondamento.
As diferenças observadas são inferiores aos desvios-padrão entre folds dos respetivos modelos com features (0,011, 0,009 e 0,010).

Para a ultima parte do challenge testei o addboostclassifier adicionei ao model.py o adaboost_logistic e inclui-a no dicionario _MODELS, alterei o config.yaml para testar o modelo e o incluir na seccao de tunning.

Modelo	                           Accuracy média	Desvio-padrão
Logistic Regression	                   0,671	        0,015
AdaBoost com Logistic Regression	   0,673	        0,016

A accuracy aumentou de 0,671 para 0,673, correspondendo a uma melhoria de apenas 0,002 esta diferença é muito pequena quando comparada com a variabilidade observada entre os folds.
Assim, os resultados não demonstram uma melhoria substancial do desempenho através da combinação de várias regressões logísticas com AdaBoost. O modelo final pode deixar de ser linear mesmo quando combina várias regressões logísticas, porque utiliza uma votação ponderada das suas previsões.

# semana 7 boosting and metrics

Esta semana, relativamente ao 06_boosting, importei no model.py os três novos modelos (GradientBoosting, HistGradientBoosting e GaussianNB) e adicionei-os ao dicionário _MODELS.
No config.yaml, atualizei o model.type e acrescentei ao tuning.search_spaces os novos modelos.

Relativamente ao 07_metrics, fiz as alterações pedidas em evaluate.py, tuning.py, results.py e main.py, o que permitiu avaliar e comparar oito modelos: Naive Bayes, Logistic Regression, Decision Tree, Random Forest, Oblique Random Forest, AdaBoost, Gradient Boosting e HistGradientBoosting.

Modelo	                ROC-AUC	     Std	Precision	Recall	 F1	   ROC-AUC(sem tuning)
Oblique Random Forest	 0.728	    0.012	  0.673	    0.570	0.617	0.685
Gradient Boosting	     0.727	    0.014	  0.686	    0.541	0.605	0.727
AdaBoost	             0.727	    0.014	  0.674	    0.573	0.619	0.723
Random Forest	         0.727	    0.016	  0.682	    0.551	0.609	0.684
HistGradientBoosting	 0.726	    0.014	  0.685	    0.531	0.597	0.717
Logistic Regression	     0.718	    0.014	  0.711	    0.451	0.550	0.718
Decision Tree	         0.712	    0.014	  0.661	    0.548	0.599	0.610
Naive Bayes	             0.701	    0.011	  0.713	    0.271	0.392	0.685

Foram comparados os resultados de ROC-AUC antes e depois do tuning, usando um modelo dummy como referência.

O modelo com melhor resultado após o tuning foi o Oblique Random Forest, com um ROC-AUC de 0,728. Gradient Boosting, AdaBoost e Random Forest apresentaram resultados muito próximos, de aproximadamente 0,727. Todos os modelos superaram o valor de referência do modelo dummy (0,500).

o AdaBoost apresenta o maior F1 (0.619), enquanto o Oblique Random Forest tem o ROC-AUC mais elevado (0.728). O Naive Bayes apresenta um recall bastante inferior (0.271), o que significa que identifica uma proporção menor dos casos positivos

Assim, o Oblique Random Forest é o modelo escolhido. No entanto os resultados são muito próximos dos obtidos pelo AdaBoost, Gradient Boosting e Random Forest, pelo que não existe uma vantagem expressiva de um único modelo. 

# semana 8 feature_selection

Esta semana implementei a funcionalidade de Feature Selection, através do VoteSelector, e adicionei o modelo XGBoost ao pipeline. Depois, executei o modelo XGBoost com o feature_selection True e False, com tuning em ambas as configurações.

O scoreboard atualizado, incluindo os resultados das duas configurações do XGBoost:

Modelo	                ROC-AUC	     Std	Precision	Recall	 F1	   ROC-AUC(sem tuning)
Oblique Random Forest	 0.728	    0.012	  0.673	    0.570	0.617	0.685
Gradient Boosting	     0.727	    0.014	  0.686	    0.541	0.605	0.727
AdaBoost	             0.727	    0.014	  0.674	    0.573	0.619	0.723
Random Forest	         0.727	    0.016	  0.682	    0.551	0.609	0.684
HistGradientBoosting	 0.726	    0.014	  0.685	    0.531	0.597	0.717
XGBoost (sem seleção)	 0.727	    0.015	  0.667	    0.580	0.620	0.724
XGBoost (com seleção)	 0.725	    0.013	  0.675	    0.558	0.611	0.721
Logistic Regression	     0.718	    0.014	  0.711	    0.451	0.550	0.718
Decision Tree	         0.712	    0.014	  0.661	    0.548	0.599	0.610
Naive Bayes	             0.701	    0.011	  0.713	    0.271	0.392	0.685


Para comparar as duas configurações, analisei também os resultados de Nested Cross-Validation em cada um dos cinco folds:

Fold	Com seleção	   Sem seleção
1	       0.724	     0.731
2	       0.703	     0.702
3	       0.730	     0.736
4	       0.733	     0.735
5	       0.734	     0.731


O modelo com Feature Selection manteve apenas 2 das 10 features originais: numeric__age e numeric__priors_count. Apesar desta redução significativa, obteve um ROC-AUC médio de 0.725, muito próximo dos 0.727 obtidos pelo modelo sem seleção de features. 

Usei o corrected_paired_test(), utilizando os resultados dos mesmos cinco folds. O teste apresentou uma diferença média de -0.0024, um valor de t de -0.793 e um p-value de 0.4723. Como o p-value é superior a 0.05, não existe evidência estatística suficiente para concluir que os dois modelos apresentam desempenhos diferentes.

Os resultados mostram que a aplicação de Feature Selection reduziu o número de features de 10 para apenas 2, mantendo um desempenho muito semelhante ao modelo original. Embora o XGBoost sem seleção tenha obtido um ROC-AUC ligeiramente superior (0.727 para 0.725), a diferença observada é pequena e não estatisticamente significativa. Assim, manter a Feature Selection pode ser uma boa opção, uma vez que permite obter um modelo mais simples e interpretável, sem uma perda relevante de desempenho preditivo.

# semana 9 beyond_compass

Escolhi a option A: prever se uma pessoa ganha mais de 50 mil USD por ano (`income_over_50k`) com o dataset Adult (`data/adult_train.csv`, 6000 linhas) e prever as 2000 pessoas novas de `data/adult_new.csv`.

Cada problema tem a sua pasta de resultados: `results` e `results_adult`, para os scoreboards não se misturarem. As previsões das pessoas novas ficam em `results_adult/predictions.csv` (colunas `id`, `probability`, `prediction`; 2000 linhas).

config_adult.yaml alterações:

- data: caminho, target `income_over_50k`, sensitive attribute `sex`, `drop_columns: [id, fnlwgt, education]` e `new_path` (o ficheiro a prever).
- diagnostics: `?` como placeholder; regras de validade próprias do Adult (`age` 17–90, `hours_per_week` 1–99, `capital_gain` e `capital_loss` ≥ 0); `workclass`, `occupation` e `native_country` listadas em `canonical_categories` (mapa vazio) para o `?` virar `NaN`; `redundant_columns: [education, fnlwgt]`.
- preprocessing: target encoder, RobustScaler, todas as colunas listadas à mão (inclui `v01` a `v25`) e `mnar_indicator_sources: [workclass]`.
- feature_selection.enabled: `true` (para ver que colunas, incluindo as `v`, ficam).
- output para ficar `results_adult`.

Decisões sobre sensitive attributes e missing values:

O sensitive attribute escolhido é `sex`. `race` não está em `numeric_features` nem em `categorical_features`, por isso não entra no modelo.

Usei apenas uma missing indicator (`workclass_was_missing`), porque os `?` de `workclass` e `occupation` são as mesmas pessoas.
Não tratei o `99999` de `capital_gain` como erro.

main.py alterações:

Lê o config do argumento da linha de comandos (por defeito `config.yaml`) e, se o config tiver `data.new_path`, prevê essas pessoas depois do refit (mesma limpeza, sem `drop_duplicate_rows`, e confirma que o número de linhas à saída é igual ao número de linhas à entrada).

Corri `main.py config_adult.yaml` e houve um crash no `fairness_report`:

- `src/evaluate.py`: `fairness_report` só compara com o score do COMPAS se existir a coluna `score_text`.
- `fairness_report` deixa de assumir a existência da coluna `score_text`.

Verificações:

- Limpeza (`check_clean.py`, que chama `clean_dataset()` com o config do Adult): `workclass` 379, `occupation` 381 e `native_country` 86 valores em falta, e nenhum `?` a sobrar, como esperado.
- COMPAS: com o `config.yaml` sem alterações, as alterações desta semana dão exatamente os mesmos resultados da Semana 8 (ROC-AUC 0.727 ± 0.015 com tuning, mesmos hiperparâmetros escolhidos e mesmas métricas por fold).
- Previsões de adult_new.csv 2000 linhas à saída, 2000 à entrada, id únicos

| what broke | why it broke | how i fixed it |
|---|---|---|
| `KeyError: 'score_text'` no `fairness_report` | assumia que o dataset tinha a coluna `score_text` do COMPAS | só compara com o score do COMPAS se a coluna existir |
| Cabeçalho do FPR dizia "did NOT reoffend" | texto escrito a pensar na reincidência | texto geral: "true label is 0, but predicted 1" |
| `?` não era tratado como missing | `_canonicalize_categories` só mexe nas colunas listadas em `canonical_categories` | listei `workclass`, `occupation` e `native_country` no config (mapa vazio) |
| `age: {min: 18}` apagaria 74 pessoas | regra de validade do COMPAS | regras próprias do Adult (`age` 17–90) |

| Run      | Modelo                              | Tuning                   | Feature selection                     | ROC-AUC           | Accuracy    | F1         |
|----------|-------------------------------------|--------------------------|---------------------------------------|-------------------|-------------|------------|
| **A**    | XGBoost (hiperparâmetros do config) | não                      | `min_votes 2`                         | **0.913 ± 0.016** | **0.864**   | **0.686**  |
| B        | XGBoost                             | sim, 5 trials, nested CV | `min_votes 1` (escolhido pelo Optuna) | 0.905 ± 0.014     | 0.857       | 0.648      |
| **Alvo** |                                     |                          |                                       | **≥ 0.90**        | **≥ 0.855** | **≥ 0.66** |

A Run A atinge os três objetivos definidos na mesma execução, superando os valores mínimos de ROC-AUC, accuracy e F1.

A Run B também cumpre os objetivos de ROC-AUC e accuracy, mas fica abaixo do valor mínimo estabelecido para o F1. Ainda assim, a nested CV apresenta uma estimativa de optimismo de apenas -0.001, o que indica uma diferença praticamente nula entre a performance estimada e a observada. Assim, o ROC-AUC de 0.905 constitui uma estimativa fiável do desempenho do modelo.

Features mantidas (8 de 36)

| Tipo | Mantidas | Não mantidas |
|---|---|---|
| Numéricas | `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week` | `v01` a `v18` |
| Categóricas | `marital_status`, `occupation`, `relationship` | `workclass`, `native_country`, `v19` a `v25` |
| Missing indicator | n/a | `workclass_was_missing` |

| Grupo | FPR | n (verdadeiros negativos) |
|---|---|---|
| Mulheres | 0.03 | 1390 |
| Homens | 0.08 | 2261 |

O `sex` não é uma feature do modelo, mas o modelo prevê `>50K` mais vezes para homens.

`results_adult/predictions.csv` tem as 2000 pessoas de `adult_new.csv`, previstas pelo modelo da Run A (refit em todo o conjunto de desenvolvimento).
A suposição mais difícil de encontrar foi a do ?: o _canonicalize_categories só trata o placeholder nas colunas listadas no config, e uma coluna esquecida ficaria com a categoria ? sem qualquer erro, por isso só a verificação dos value_counts a apanha. O KeyError do fairness_report foi o único crash em execução, e a correção mantém o COMPAS idêntico. O que levou aos alvos foi sobretudo a limpeza e o config: o XGBoost com os hiperparâmetros por defeito já os atinge. O tuning com apenas 5 trials não melhorou (0.905 contra 0.913) e deixou o F1 abaixo do alvo, o que sugere que seria preciso mais trials para que compensasse.

# semana 10 missing_values e 11_deployment

Em `preprocessing.py`, o pipeline passou a aplicar scaling às variáveis numéricas antes da imputação. A estratégia é definida em `config.yaml` através de `preprocessing.imputation.numeric_strategy`

Comparei cinco imputers com o mesmo XGBoost, os mesmos cinco stratified folds e o tuning desligado (False). O final test set ficou reservado. Usei `median` como baseline e corri `corrected_paired_test()` sobre os valores de ROC-AUC dos mesmos folds de validação.

| Imputer | Run ID | Mean ROC-AUC ± SD | Δ vs. median | Corrected p-value |
|---|---|---:|---:|---:|
| `median` | `20261120_193112` | 0.72399 ± 0.01743 | — | — |
| `mean` | `20261120_193152` | 0.72353 ± 0.01651 | −0.00047 | 0.743 |
| `knn` | `20261120_193237` | 0.72519 ± 0.01532 | +0.00119 | 0.632 |
| `iterative` | `20261120_193348` | 0.72375 ± 0.01791 | −0.00024 | 0.933 |
| `random_forest` | `20261120_193436` | 0.72318 ± 0.01562 | −0.00081 | 0.545 |

O `knn` obteve o ROC-AUC médio mais alto e superou a mediana em 4 dos 5 folds, com um ganho médio de 0.00119. Nenhuma das diferenças face à mediana foi estatisticamente significativa ao nível de 5%. Escolhi manter `median` pela simplicidade e pelo menor custo de treino.

Após o treino final, `main.py` guarda o pipeline completo em `model.joblib`. O `predict.py` carrega esse pipeline e aplica as mesmas regras de limpeza a um novo CSV, sem voltar a treinar o modelo.

O comando (python predict.py data/compas_new_people.csv predictions.csv) gerou 20 previsões, uma por cada linha do CSV de entrada. O ficheiro `predictions.csv` contém três colunas:

`id` — identificador da linha de entrada.
`probability` — probabilidade estimada da classe positiva.
`prediction` — classe `0` ou `1`, calculada com um threshold de `0.5`.

# semana 11 outliers

Esta semana dicionei o `AnomalyScore` a `src/preprocessing.py`. Em `src/model.py`, coloquei este passo depois do preprocessing e antes da feature selection. Em cada fold, o detetor é ajustado apenas com os dados de treino e acrescenta o `anomaly_score` como feature.

A secção `anomaly` do `config.yaml` permite escolher entre `none`, `isolation_forest` e `lof`. O `main.py` regista o método na coluna `anomaly` do `runs.csv` e no scoreboard. O `src/results.py` inclui o método na primeira linha do relatório de cada run.

Inicialmente, esperava que as pessoas no top 5% por `anomaly_score` tivessem uma taxa de reincidência superior à dos restantes 95%, porque algumas combinações de idade e antecedentes poderiam estar associadas a maior reincidência. Esperava também que ao incluir o `anomaly_score` melhorasse o ROC-AUC do XGBoost, pois o score resume esses padrões numa única feature que o modelo poderia aproveitar.

Comparei a taxa de reincidência das 289 pessoas (5%) com maior `anomaly_score` com a das 5 482 (95%) restantes:

| Anomaly method | Top 5% | Remaining 95% | Difference |
|---|---:|---:|---:|
| Isolation Forest | 58.48% (169/289) | 44.36% (2 432/5 482) | +14.11 pp |
| LOF | 51.56% (149/289) | 44.73% (2 452/5 482) | +6.83 pp |

O top 5% teve maior reincidência observada com ambos. A diferença foi maior com Isolation Forest.

Corri as três configurações com o mesmo modelo (XGBoost), median imputation, os mesmos cinco stratified folds e tuning desligado (false). Usei `corrected_paired_test()` para comparar o ROC-AUC de cada detetor com o baseline (`none`), fold a fold.

| Anomaly method | Run ID | Mean ROC-AUC ± SD | Δ vs. baseline | Folds won | Corrected p-value |
|---|---|---:|---:|---:|---:|
| `none` | `20261125_114703` | 0.72399 ± 0.01743 | — | — | — |
| `isolation_forest` | `20261125_114811` | 0.72040 ± 0.01528 | −0.00360 | 1/5 | 0.294 |
| `lof` | `20261125_114905` | 0.71860 ± 0.01841 | −0.00539 | 0/5 | 0.022 |

O none teve o melhor ROC-AUC médio. A diferença do Isolation Forest foi inconclusiva, enquanto o LOF baixou o ROC-AUC nos cinco folds, com p = 0.022 no teste individual. A maior reincidência observada no top 5% não se traduziu num ganho de ROC-AUC quando acrescentei o score ao XGBoost. Uma possível explicação é que o XGBoost já tinha acesso às variáveis usadas para calcular o `anomaly_score`, pelo que o score pode ter acrescentado pouca informação útil. Com base nos resultados, escolhi manter `anomaly.method: "none"`.

Para alem disso testei o pipeline com LOF ativo e verifiquei que o `predictions.csv` manteve as 20 linhas de dados, com as colunas `id`, `probability` e `prediction` ao correr novamnete `python predict.py data/compas_new_people.csv predictions.csv`

