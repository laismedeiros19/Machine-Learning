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

## Como executar: comece por aqui

Para testar o projeto, siga os passos **1 a 7** abaixo. O modelo já está treinado e o dataset já acompanha o projeto.

### 1. Instale e abra o Docker

Instale o Docker seguindo as instruções para seu sistema: https://docs.docker.com/get-started/get-docker/.

- **Windows ou macOS:** instale o Docker Desktop e abra o aplicativo. Aguarde ele indicar que está funcionando.
- **Linux:** instale o Docker Engine e o plugin Docker Compose pelo guia oficial e inicie o serviço Docker.

A primeira execução precisa de internet para baixar a imagem e as bibliotecas. Para este caminho, não é necessário instalar Python no computador.

### 2. Baixe os arquivos do projeto

1. Abra https://github.com/laismedeiros19/Machine-Learning.
2. Clique no botão verde **Code**.
3. Clique em **Download ZIP**.
4. Localize o arquivo baixado e extraia todo o conteúdo.
5. Abra a pasta **Machine-Learning-main**. Ela deve conter `README.md`, `Dockerfile`, `docker-compose.yml` e `modelo.pkl`.

Se recebeu o ZIP da entrega, extraia-o e use a pasta **projeto_hoteis**. Não execute o projeto de dentro do arquivo ZIP.

**Alternativa para quem já usa Git:**

```bash
git clone https://github.com/laismedeiros19/Machine-Learning.git
cd Machine-Learning
```

Nesse caso, a pasta se chama **Machine-Learning**. Continue no passo 3.

### 3. Abra o terminal na pasta correta

Uma forma simples é usar o VS Code:

1. Clique em **Arquivo → Abrir Pasta** e selecione a pasta extraída no passo 2.
2. Clique em **Terminal → Novo Terminal**.
3. O terminal deve abrir nessa pasta. Para conferir, execute:

```bash
ls
```

No Prompt de Comando do Windows, use `dir` em vez de `ls`. A lista deve mostrar `Dockerfile` e `docker-compose.yml`. Se não mostrar, abra a pasta que contém esses arquivos.

Quem não usa VS Code pode abrir o terminal do sistema e usar `cd` para entrar na pasta extraída.

### 4. Inicie a aplicação

Copie este comando, cole no terminal e pressione Enter:

```bash
docker compose up -d --build
```

Aguarde o comando terminar. Na primeira vez, o download e a instalação podem levar alguns minutos. Esse comando cria o ambiente e inicia a aplicação em segundo plano.

O PDF usa a escrita abaixo, que funciona em instalações que oferecem o comando com hífen:

```bash
docker-compose up -d --build
```

Use **uma** dessas formas, conforme sua instalação. Nos passos seguintes, os exemplos usam `docker compose`.

Confira se a aplicação iniciou:

```bash
docker compose ps
```

Espere aparecer **Up** e **healthy**. Se aparecer `starting`, aguarde alguns segundos e execute o comando novamente.

### 5. Teste pelo navegador

1. Abra o navegador.
2. Digite **http://localhost:5000/** na barra de endereço.
3. O formulário já vem preenchido com uma reserva de exemplo.
4. Clique em **Prever cancelamento**.
5. A resposta esperada é **Não cancelamento previsto**, com probabilidade de cancelamento de **20,42%**.
6. Altere os campos e clique novamente para testar outras reservas.
7. Clique em **Restaurar exemplo** para voltar aos valores iniciais.

A classe pode continuar igual depois de alterar um campo. A previsão é uma estimativa, não uma garantia de que a reserva será mantida ou cancelada.

### 6. Teste a API pelo terminal (opcional)

Este teste atende ao endpoint exigido na avaliação e funciona sem usar o formulário. Mantenha a aplicação iniciada no passo 4 e abra outro terminal **na mesma pasta do projeto**.

O arquivo `exemplo_reserva.json` já contém uma reserva completa. Em Linux/macOS, execute:

```bash
curl -i -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @exemplo_reserva.json
```

No Windows, execute:

```powershell
curl.exe -i -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "@exemplo_reserva.json"
```

A resposta deve ter **HTTP 200** e este conteúdo JSON (a ordem dos campos pode variar):

```json
{
  "booking_status": "Not_Canceled",
  "classe": 0,
  "probabilidade_cancelamento": 0.2042
}
```

A classe **0** significa não cancelamento e **1** significa cancelamento. A probabilidade `0.2042` corresponde a **20,42%**.

Para testar outra reserva, abra `exemplo_reserva.json`, altere os valores, salve e execute o comando novamente. O arquivo original tem este conteúdo:

```json
{
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
}
```

A API recebe os 17 atributos usados no modelo. Envie números nas colunas numéricas, contagens inteiras e textos nas categorias. Não envie `Booking_ID` nem `booking_status`.

Entradas inválidas retornam **HTTP 400** com uma mensagem em JSON. Conteúdo sem `Content-Type: application/json` retorna **HTTP 415**. A API verifica campos, tipos, data válida, valores não negativos, pelo menos um hóspede e uma noite.

### 7. Encerre a aplicação

Quando terminar, execute na pasta do projeto:

```bash
docker compose down
```

Para usar novamente, repita o passo 4. O dataset e o modelo continuam na pasta.

## Se algo não funcionar

| O que apareceu | O que fazer |
|---|---|
| `docker: command not found` ou comando não reconhecido | Confira a instalação do Docker no passo 1 e reabra o terminal. |
| Erro ao conectar ao serviço Docker | Abra o Docker Desktop ou confira se o serviço Docker está iniciado no Linux. |
| `permission denied` ao acessar o Docker | Confira as permissões do usuário conforme o guia de instalação do Docker para seu sistema. |
| `no configuration file provided` | Abra o terminal na pasta que contém `docker-compose.yml`. |
| Porta 5000 já está em uso | Encerre a API manual com Ctrl + C no terminal onde ela está aberta ou use a porta alternativa abaixo. |
| Página não abre | Confira `docker compose ps` e os logs com o comando abaixo. |
| `405 Method Not Allowed` ao abrir `/predict` no navegador | Para o formulário, abra somente `http://localhost:5000/`. A rota `/predict` recebe POST com JSON, como no passo 6. |

Para ver mensagens da aplicação:

```bash
docker compose logs api
```

### Como usar a porta 5001

Em Linux/macOS:

```bash
PORTA_HOST=5001 docker compose up -d --build
```

No PowerShell do Windows:

```powershell
$env:PORTA_HOST="5001"
docker compose up -d --build
```

Abra **http://localhost:5001/**. No comando curl, troque também `5000` por `5001`.

## Alternativa: executar a aplicação sem Docker

Use este caminho se quiser executar diretamente com Python. Não é necessário seguir esta seção se o Docker já funcionou.

### 1. Prepare o computador e a pasta

Instale **Python 3.12** pelo site https://www.python.org/downloads/. No Windows, marque a opção de adicionar Python ao PATH durante a instalação.

Baixe e extraia o projeto e abra o terminal na pasta, como nos passos 2 e 3 do guia principal. Encerre o Docker com `docker compose down` caso esteja usando a porta 5000.

### 2. Crie o ambiente Python

Em Linux/macOS:

```bash
python3.12 -m venv .venv
```

No Windows:

```powershell
py -3.12 -m venv .venv
```

### 3. Instale as bibliotecas

Em Linux/macOS:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

No Windows:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Inicie a API

Em Linux/macOS:

```bash
.venv/bin/python app.py
```

No Windows:

```powershell
.venv\Scripts\python.exe app.py
```

Deixe esse terminal aberto. Quando aparecer `Running on http://127.0.0.1:5000`, abra **http://localhost:5000/** e teste o formulário. Para testar pelo curl, use outro terminal e siga o passo 6 do guia principal.

### 5. Encerre

No terminal que executa a API, pressione **Ctrl + C**.

## Como executar o notebook e reproduzir o treinamento

O modelo exportado já permite testar a aplicação. Siga esta seção se quiser refazer o carregamento, a preparação, o treinamento, a avaliação e a exportação.

### 1. Crie o ambiente

Instale Python 3.12, abra o terminal na pasta do projeto e crie `.venv`, seguindo os passos 1 e 2 da execução sem Docker. Confira se existe `dataset/Hotel Reservations.csv`.

### 2. Instale as dependências do notebook

Em Linux/macOS:

```bash
.venv/bin/python -m pip install -r requirements-treinamento.txt
```

No Windows:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-treinamento.txt
```

### 3. Abra o notebook

Em Linux/macOS:

```bash
.venv/bin/python -m jupyter notebook notebook.ipynb
```

No Windows:

```powershell
.venv\Scripts\python.exe -m jupyter notebook notebook.ipynb
```

O Jupyter abre no navegador. Se não abrir automaticamente, copie no navegador o endereço mostrado no terminal.

### 4. Execute as células

1. Selecione o kernel Python desse ambiente.
2. Execute as células em ordem, começando pela questão 1, com **Shift + Enter**.
3. Aguarde as etapas de validação e treinamento terminarem.
4. Confira as tabelas, gráficos e interpretações da questão 4.
5. A questão 5 salva novamente `modelo.pkl` com a Pipeline completa.

Não altere os parâmetros ou use o teste para procurar métricas melhores ao reproduzir o resultado entregue.

### 5. Use o modelo gerado

Reinicie a aplicação local para carregar o novo arquivo. Se usar Docker, execute `docker compose up -d --build` para incluir o modelo novo na imagem.

A aplicação carrega a Pipeline pronta e não realiza treinamento durante a inicialização ou a cada requisição.

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
├── exemplo_reserva.json
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
