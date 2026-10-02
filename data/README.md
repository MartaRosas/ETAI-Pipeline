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
 
