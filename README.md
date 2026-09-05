# RAG - Retrieval-Augmented Generation

## Sobre o projeto

Este projeto foi desenvolvido de forma simples e didática com o objetivo de entender, na prática, como funciona a construção de um sistema de **RAG (Retrieval-Augmented Generation)**, também conhecido como **Geração Aumentada via Recuperação** todo ensinamento aprendido neste curso foi realizado no Canal [TeoMeWhy](https://www.youtube.com/watch?v=HGi3V2S2Y7s&list=PLvlkVRRKOYFTijx_2mJBv7u2F21VD-Zgb&index=1), minha base de dados foram arquivos que explicam de forma teórica sobre Aprendizado de maquina como por exemplo : "o que é overfitting?, data lekeage, Entropia" e etc..

A ideia principal não foi construir uma solução complexa ou pronta para produção, mas sim explorar os principais conceitos e etapas necessários para criar um sistema capaz de:

- Armazenar documentos em formato de texto
- Dividir os documentos em pequenos trechos chamados Chunks
- Transformar os textos em representações vetoriais utilizando Embeddings
- Armazenar e consultar os vetores em um banco de dados vetorial
- Utilizar o Qdrant para realizar a busca dos documentos mais relevantes
- Utilizar diferentes estratégias de recuperação, como Dense, Sparse e ColBERT
- Utilizar o Groq para auxiliar na geração das respostas
- Criar um modelo simples de Guardrails
- Registrar e acompanhar o modelo utilizando MLflow
- Disponibilizar o modelo através de uma API desenvolvida com Flask

O projeto foi desenvolvido principalmente como uma forma de estudo e aprendizado sobre a arquitetura de aplicações baseadas em RAG.

---

## Objetivos

Os principais objetivos deste projeto foram compreender os conceitos fundamentais envolvidos em uma aplicação RAG.

Entre eles:

- Entender o que são **Chunks**
- Entender o conceito de **Embeddings**
- Compreender a diferença entre modelos de recuperação **Dense**, **Sparse** e **ColBERT**
- Entender o funcionamento de um **banco de dados vetorial**
- Aprender como realizar a ingestão de documentos no **Qdrant**
- Aprender como realizar buscas vetoriais utilizando a API do Qdrant
- Recuperar informações relevantes de acordo com uma pergunta
- Utilizar um modelo de linguagem para gerar uma resposta baseada nos documentos recuperados
- Entender o conceito de **Guardrails**
- Criar um modelo simples para controlar as respostas
- Registrar o modelo utilizando **MLflow**
- Criar uma API utilizando **Flask**
- Integrar todas essas etapas em um fluxo simples de RAG

---

# O que é RAG?

RAG significa **Retrieval-Augmented Generation**.

A ideia principal é combinar duas etapas:

1. **Retrieval (Recuperação)**  
   Buscar informações relevantes em uma base de conhecimento.

2. **Generation (Geração)**  
   Utilizar um modelo de linguagem para gerar uma resposta utilizando as informações recuperadas.

De forma simplificada:

```text
Pergunta do usuário
        |
        v
Transformação da pergunta
        |
        v
Busca no banco vetorial
        |
        v
Documentos relevantes
        |
        v
Modelo de linguagem
        |
        v
Resposta
```

Isso permite que o modelo utilize informações presentes em uma base de conhecimento específica, em vez de depender apenas do conhecimento utilizado durante seu treinamento.

---

# Conceitos estudados

## Chunks

Chunks são pequenos pedaços de um documento.

Em vez de armazenar um documento inteiro como uma única informação, ele pode ser dividido em partes menores.

Por exemplo:

```text
Documento original
        |
        +-- Chunk 1
        +-- Chunk 2
        +-- Chunk 3
        +-- Chunk 4
```

A divisão em Chunks facilita a recuperação de informações específicas durante a busca.

---

## Embeddings

Embeddings são representações numéricas de textos.

Um texto é transformado em um vetor que representa suas características semânticas.

Por exemplo:

```text
"Como funciona Machine Learning?"
                |
                v
        Modelo de Embedding
                |
                v
[0.12, -0.45, 0.87, 0.31, ...]
```

Esses vetores podem ser armazenados em um banco de dados vetorial e utilizados para encontrar textos semanticamente semelhantes.

---

# Recuperação de informações

Neste projeto foram estudadas diferentes abordagens de recuperação.

## Dense Retrieval

A busca densa utiliza embeddings para representar semanticamente os textos.

A consulta também é transformada em um vetor e o sistema procura os vetores mais semelhantes.

```text
Pergunta
   |
   v
Embedding
   |
   v
Busca por similaridade
   |
   v
Chunks relevantes
```

Essa abordagem é bastante útil quando queremos encontrar conteúdos semanticamente relacionados mesmo quando as palavras utilizadas são diferentes.

---

## Sparse Retrieval

A recuperação sparse utiliza representações esparsas e dá maior importância à correspondência entre termos.

É uma abordagem que pode ser interessante quando palavras específicas possuem grande importância na consulta.

---

## ColBERT

O ColBERT é uma abordagem de recuperação que trabalha com representações mais detalhadas dos tokens do texto.

Em vez de representar todo o documento utilizando apenas um vetor, ele permite uma comparação mais detalhada entre os tokens da consulta e do documento.

Neste projeto, o objetivo principal foi compreender o conceito e sua utilização dentro de sistemas de recuperação.

---

# Banco de dados vetorial

Para armazenar os embeddings foi utilizado o **Qdrant**.

O Qdrant é um banco de dados vetorial que permite armazenar vetores e realizar buscas por similaridade.

O fluxo utilizado no projeto pode ser representado da seguinte maneira:

```text
Arquivo TXT
    |
    v
Divisão em Chunks
    |
    v
Embeddings
    |
    v
Qdrant
    |
    v
Busca por similaridade
```

---

# Ingestão no Qdrant

A ingestão é responsável por preparar os documentos e armazená-los no banco vetorial, feito pelo **01_ingest.py**

De forma simplificada:

```text
Documentos
    |
    v
Leitura dos arquivos
    |
    v
Divisão em Chunks
    |
    v
Geração dos Embeddings
    |
    v
Criação dos vetores
    |
    v
Inserção no Qdrant
```

Cada Chunk armazenado possui informações que permitem posteriormente recuperar o conteúdo relacionado à pergunta realizada pelo usuário.

---

# Busca através da API do Qdrant

Depois que os documentos são armazenados, uma pergunta pode ser enviada para o sistema, feitas pela **02_query.py**

A pergunta é transformada em uma representação adequada para busca e enviada ao Qdrant.

O Qdrant retorna os vetores ou documentos mais semelhantes.

Exemplo conceitual:

```text
Pergunta:
"O que é Machine Learning?"

              |
              v

        Busca no Qdrant

              |
              v

Resultados encontrados:

Chunk 1
Chunk 2
Chunk 3
```

Esses resultados são utilizados como contexto para a próxima etapa.

---

# Groq

Após recuperar os documentos relevantes, o contexto é enviado para um modelo de linguagem através do **Groq**.

O modelo utiliza as informações recuperadas para gerar uma resposta.

O fluxo pode ser representado como:

```text
Pergunta
   |
   v
Qdrant
   |
   v
Chunks relevantes
   |
   v
Contexto
   |
   v
Groq
   |
   v
Resposta
```

O objetivo é fazer com que a resposta seja baseada principalmente nas informações recuperadas da base de conhecimento.

---

# Guardrails

Outro objetivo do projeto foi compreender o conceito de **Guardrails**.

Guardrails podem ser utilizados para estabelecer regras e controles sobre o comportamento de uma aplicação baseada em modelos de linguagem, feito pelo **train.py**

Neste projeto foi criado um modelo simples com o objetivo de demonstrar como uma camada de controle pode ser adicionada ao fluxo da aplicação.

De forma simplificada:

```text
Pergunta
   |
   v
RAG
   |
   v
Resposta gerada
   |
   v
Guardrails
   |
   v
Resposta final
```

O Guardrails pode atuar como uma camada adicional para verificar ou controlar determinadas respostas antes que elas sejam retornadas ao usuário.

---

# MLflow

O **MLflow** foi utilizado para compreender o processo de registro e gerenciamento do modelo criado para o Guardrails.

O fluxo estudado foi:

```text
Criação do modelo
       |
       v
Treinamento
       |
       v
Avaliação
       |
       v
MLflow
       |
       v
Registro do modelo
```

A utilização do MLflow permitiu entender conceitos relacionados a:

- Registro de modelos
- Versionamento
- Experimentos
- Rastreamento
- Gerenciamento do ciclo de vida do modelo

---

# API Flask

Depois de construir o fluxo do projeto, foi criada uma API simples utilizando **Flask**.

A API recebe uma requisição `POST` contendo uma pergunta.

Exemplo:

```http
POST /predict
```

Com um JSON semelhante a:

```json
{
    "question": "O que é Machine Learning?"
}
```

A aplicação processa a pergunta e retorna uma resposta baseada no fluxo desenvolvido.

Exemplo conceitual:

```json
{
    "answer": "Machine Learning é uma área da inteligência artificial..."
}
```

---

# Arquitetura do projeto

O fluxo completo pode ser representado da seguinte maneira:

```text
                  DOCUMENTOS
                      |
                      v
                Leitura dos TXT
                      |
                      v
                    Chunks
                      |
                      v
                  Embeddings
                      |
                      v
                    Qdrant
                      |
                      |
                +-----+-----+
                |           |
                v           v
             Dense       Sparse
                |           |
                +-----+-----+
                      |
                      v
                    ColBERT
                      |
                      v
             Recuperação dos dados
                      |
                      v
                   Contexto
                      |
                      v
                    Groq
                      |
                      v
               Resposta gerada
                      |
                      v
                  Guardrails
                      |
                      v
                    MLflow
                      |
                      v
                  Modelo/API
                      |
                      v
                    Flask
                      |
                      v
                 Resposta JSON
```

---



# Tecnologias utilizadas

As principais tecnologias utilizadas no projeto foram:

- Python
- Qdrant
- Groq
- MLflow
- Flask
- Embeddings
- Vector Database
- Dense Retrieval
- Sparse Retrieval
- ColBERT
- Guardrails
- API REST


# O que aprendi com o projeto

Este projeto foi desenvolvido principalmente como uma experiência prática de aprendizado.

Os principais conceitos estudados foram:

- Funcionamento de um pipeline RAG
- Divisão de documentos em Chunks
- Geração e utilização de Embeddings
- Busca semântica
- Recuperação Dense
- Recuperação Sparse
- Conceito de ColBERT
- Funcionamento de bancos vetoriais
- Ingestão de dados no Qdrant
- Busca utilizando a API do Qdrant
- Utilização de modelos de linguagem através do Groq
- Conceito de Guardrails
- Registro de modelos utilizando MLflow
- Desenvolvimento de APIs REST utilizando Flask
- Integração entre diferentes componentes de uma aplicação de Inteligência Artificial

---


## Conclusão 

Apesar de toda criação ter sido ótima para o aprendizado ainda existem lacunas para serem preenchidas, 
por exemplo em alguns momentos a API não responde quando é questionada sobre "o que é machie learning?" ou as vezes
ela responde quando é colocada uma letra maiuscula ou com a uma letra maiúscula, são esses pontos que podem ser estudados e melhorados.


---

## Próximos passos

Algumas possibilidades de evolução do projeto são:

- Melhorar a estratégia de Chunking
- Comparar diferentes modelos de Embeddings
- Avaliar Dense, Sparse e ColBERT utilizando métricas
- Implementar Reranking
- Criar uma avaliação automática das respostas
- Adicionar observabilidade ao pipeline
- Melhorar os Guardrails
- Utilizar modelos maiores e mais especializados
- Criar uma interface web para interação com o RAG
- Utilizar Docker
- Estruturar a aplicação para ambiente de produção

## Referência

Projeto RAGIA do :  [TeoMeWhy](https://www.youtube.com/watch?v=HGi3V2S2Y7s&list=PLvlkVRRKOYFTijx_2mJBv7u2F21VD-Zgb&index=1) 
