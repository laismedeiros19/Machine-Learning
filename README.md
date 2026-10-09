# Previsão de cancelamento de reservas de hotel

**Aluna:** Lais Gabrielly Vital Medeiros  
**Disciplina:** Aprendizado de Máquina — Avaliação 1, 2026/2

Desenvolvi um projeto de **classificação binária** para prever se uma reserva será cancelada. Comparei três algoritmos, exportei o modelo escolhido e criei uma API Flask com uma interface para testar as previsões pelo navegador.

## Dataset

Usei o **Hotel Reservations Dataset**, publicado por Ahsan Raza no Kaggle:

https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset

O arquivo utilizado acompanha o projeto em `dataset/Hotel Reservations.csv`. Para obtê-lo novamente, abra o endereço acima, clique em Download e extraia o CSV nessa pasta. O Kaggle pode solicitar uma conta para o download.

Cada linha representa uma reserva. A base original tem **36.275 registros e 19 colunas**. Depois da limpeza, mantive **36.160 registros**. Não usei amostragem nem dados sintéticos.

O alvo é `booking_status`: `Not_Canceled` foi convertido em **0**, e `Canceled` em **1**. Excluí `Booking_ID` dos atributos porque é um identificador.

### Atributos utilizados

| Atributo | Descrição |
|---|---|
| `no_of_adults` | Número de adultos |
| `no_of_children` | Número de crianças |
| `no_of_weekend_nights` | Noites de fim de semana |
| `no_of_week_nights` | Noites durante a semana |
| `required_car_parking_space` | Estacionamento: 0 ou 1 |
| `lead_time` | Antecedência da reserva em dias |
| `arrival_year` | Ano da chegada |
| `arrival_month` | Mês da chegada |
| `arrival_date` | Dia do mês da chegada |
| `repeated_guest` | Hóspede recorrente: 0 ou 1 |
| `no_of_previous_cancellations` | Cancelamentos anteriores |
| `no_of_previous_bookings_not_canceled` | Reservas anteriores não canceladas |
| `avg_price_per_room` | Preço médio por quarto |
| `no_of_special_requests` | Quantidade de pedidos especiais |
| `type_of_meal_plan` | Plano de alimentação |
| `room_type_reserved` | Tipo de quarto |
| `market_segment_type` | Segmento da reserva |

## Preparação e treinamento

Conferi os tipos, valores ausentes, duplicatas, categorias e valores extremos. Não encontrei valores ausentes ou IDs repetidos. Mantive reservas com IDs diferentes mesmo quando os atributos eram iguais, pois podem ser reservas distintas.

Removi **37 datas inválidas e 78 reservas sem noites**, totalizando 115 registros. Mantive preços iguais a zero por não ter evidência de que eram erros. Também mantive valores extremos válidos e não apliquei balanceamento artificial.

Separei **28.928 reservas para treino** e **7.232 para teste** com `StratifiedGroupKFold`, tentando preservar a proporção das classes. Agrupei reservas com os mesmos atributos para impedir que perfis idênticos aparecessem nos dois conjuntos. O alvo e o ID não participaram da definição dos grupos.

Comparei os modelos nas mesmas cinco divisões de validação dentro do treino, também respeitando os grupos. Usei o F1 da classe cancelada como critério principal. O teste foi reservado para avaliar o modelo escolhido, sem ajustar parâmetros com seus resultados.

Usei `Pipeline` e `ColumnTransformer` para manter o pré-processamento junto do modelo. Preparei imputação pela mediana e pela categoria mais frequente, codificação com `OneHotEncoder` e padronização apenas para a regressão logística. As transformações foram aprendidas somente nos dados de treino de cada divisão.

Testei estas configurações iniciais:

- **Regressão logística:** `C=1.0`, `solver="lbfgs"`, `max_iter=2000`.
- **Árvore de decisão:** `max_depth=8`, `min_samples_leaf=20`, critério Gini.
- **Random Forest:** 200 árvores, `max_depth=12`, `min_samples_leaf=5` e `max_features="sqrt"`.

Usei `random_state=42`. Não realizei uma busca exaustiva de hiperparâmetros.

## Resultados

### Validação no treino

As métricas abaixo são médias das cinco divisões. Precisão, recall e F1 se referem à classe cancelada.

| Algoritmo | F1 | Precisão | Recall | Acurácia |
|---|---:|---:|---:|---:|
| Regressão logística | 68,34% | 74,18% | 63,44% | 80,73% |
| Árvore de decisão | 73,68% | 77,65% | 70,16% | 83,57% |
| Random Forest | **75,07%** | **83,47%** | 68,29% | **85,13%** |

Escolhi o **Random Forest pelo maior F1 médio**. A árvore teve recall maior, mas o Random Forest teve precisão maior. A diferença de F1 entre os dois melhores foi pequena; isso não garante superioridade em todos os cenários.

### Avaliação final no teste

| Métrica | Resultado |
|---|---:|
| Acurácia | 84,28% |
| Precisão | 80,49% |
| Recall | 68,80% |
| F1 | 74,19% |
| ROC/AUC | 0,908 |

O modelo detectou aproximadamente 69 de cada 100 cancelamentos reais. Cerca de 80 de cada 100 alertas de cancelamento estavam corretos. A AUC indica capacidade de separar as classes em diferentes limites de decisão; não é o percentual de acertos.

Na matriz de confusão, acertei **4.461 reservas não canceladas** e **1.634 canceladas**. Houve **396 alertas falsos** e **741 cancelamentos não detectados**. Os gráficos e as interpretações estão na questão 4 do notebook.

A base contém registros de 2017 e 2018. A divisão avalia perfis diferentes, sem simular períodos futuros. Não encontrei documentação suficiente para confirmar o momento de atualização de preço e pedidos especiais. A previsão é uma estimativa e pode apresentar resultados diferentes em outros hotéis ou períodos.

## Como obter o projeto

### Pelo GitHub, usando Git

Com Git instalado, abra um terminal e execute:

```bash
git clone https://github.com/laismedeiros19/Machine-Learning.git
cd Machine-Learning
```

Essa pasta contém os arquivos necessários, incluindo o dataset e o modelo exportado. Execute os comandos Docker da próxima seção dentro dela.

### Sem Git, baixando o ZIP

1. Acesse https://github.com/laismedeiros19/Machine-Learning.
2. Clique em **Code → Download ZIP**.
3. Extraia o arquivo e abra um terminal dentro da pasta **Machine-Learning-main**, onde estão `Dockerfile` e `docker-compose.yml`.

Se recebeu o ZIP da entrega diretamente, extraia-o e abra um terminal na pasta **projeto_hoteis**. Depois siga os mesmos comandos Docker abaixo.

## Execução com Docker

É necessário ter **Docker e Docker Compose** instalados e o serviço Docker em execução. A primeira construção precisa de internet para baixar a imagem Python e as dependências.

Abra um terminal na pasta do projeto e execute o comando solicitado na avaliação:

```bash
docker-compose up -d --build
```

Em instalações recentes, o comando equivalente é:

```bash
docker compose up -d --build
```

Abra **http://localhost:5000/** para testar o formulário. Ele já vem preenchido com um exemplo. Clique em **Prever cancelamento** para consultar a API.

Para conferir o estado, os logs e encerrar a aplicação:

```bash
docker compose ps
docker compose logs api
docker compose down
```

Se estiver executando `app.py` manualmente, encerre-o com **Ctrl + C** antes de iniciar o Docker para liberar a porta 5000. Como alternativa, em Linux/macOS:

```bash
PORTA_HOST=5001 docker compose up -d --build
```

Nesse caso, use **http://localhost:5001/** e substitua a porta no curl. No PowerShell, defina `$env:PORTA_HOST="5001"` antes do comando Compose.

A imagem inclui a API, o modelo exportado e a interface. O contêiner **não realiza treinamento** na inicialização ou nas requisições.

## Execução local e reprodução do treinamento

Use **Python 3.12**. Os comandos abaixo são para Linux/macOS, a partir da pasta do projeto:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

No Windows, crie o ambiente com `py -3.12 -m venv .venv` e use `.venv\Scripts\python.exe` no lugar de `.venv/bin/python`.

Para reproduzir o fluxo completo, instale também as dependências do notebook:

```bash
.venv/bin/python -m pip install -r requirements-treinamento.txt
.venv/bin/python -m jupyter notebook notebook.ipynb
```

Selecione o kernel desse ambiente e execute as células em ordem, a partir da questão 1. O CSV deve estar no caminho indicado. O notebook contempla carregamento, preparação, treinamento, avaliação e exportação; a questão 5 grava novamente `modelo.pkl`. Reinicie a aplicação depois de gerar um novo modelo.

## API de inferência

**Endpoint:** `POST /predict`  
**Content-Type:** `application/json`

Envie um objeto com os 17 atributos da tabela. Os atributos numéricos devem ser números, as contagens inteiras e as categorias textos. Não envie `Booking_ID` nem `booking_status`. A API verifica campos, tipos, data válida, valores não negativos, presença de hóspedes e pelo menos uma noite.

### Exemplo completo de requisição

Com a aplicação em execução, abra outro terminal:

```bash
curl -X POST http://localhost:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "no_of_adults": 2,
    "no_of_children": 0,
    "no_of_weekend_nights": 1,
    "no_of_week_nights": 2,
    "required_car_parking_space": 0,
    "lead_time": 60,
    "arrival_year": 2018,
    "arrival_month": 10,
    "arrival_date": 15,
    "repeated_guest": 0,
    "no_of_previous_cancellations": 0,
    "no_of_previous_bookings_not_canceled": 0,
    "avg_price_per_room": 100.0,
    "no_of_special_requests": 1,
    "type_of_meal_plan": "Meal Plan 1",
    "room_type_reserved": "Room_Type 1",
    "market_segment_type": "Online"
  }'
```

No PowerShell, use `curl.exe` ou uma ferramenta equivalente para enviar esse JSON.

### Resposta obtida

```json
{
  "booking_status": "Not_Canceled",
  "classe": 0,
  "probabilidade_cancelamento": 0.2042
}
```

A classe 0 indica não cancelamento; 1 indica cancelamento. A probabilidade é uma estimativa do modelo. Usei o limite padrão de 0,5 para definir a classe.

Uma entrada inválida retorna HTTP 400 com `{"erro": "mensagem"}`. Um conteúdo sem o tipo JSON retorna HTTP 415. O formulário envia os dados ao mesmo endpoint e exibe o resultado.

## Estrutura do projeto

```text
projeto/
├── README.md
├── dataset/
│   └── Hotel Reservations.csv
├── notebook.ipynb
├── modelo.pkl
├── app.py
├── requirements.txt
├── requirements-treinamento.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── app.js
```

## Verificação realizada

Construí a imagem Docker e iniciei o contêiner de teste. A verificação de saúde passou, a página e seus arquivos responderam com HTTP 200 e o endpoint retornou o JSON do exemplo acima. Também confirmei que o modelo recarregado preservou as classes e probabilidades do modelo em memória.
