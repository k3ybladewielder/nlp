# Stanford XCS224U: Natural Language Understanding

## Course Overview, Part 1
...

## Course Overview, Part 2
...

## Static Vector Representation of Words
### 1. Feature-based (sparse):
   - **Explicação:** Este método baseia-se em representar palavras como vetores de características, onde cada característica representa uma propriedade específica da palavra. Cada palavra é associada a um vetor de dimensões discretas, onde apenas algumas dessas dimensões (ou características) são ativadas, tornando a representação esparsa.
   - **Exemplo:** Em um contexto de processamento de linguagem natural (NLP), uma palavra poderia ser representada por um vetor de características, onde cada dimensão representa a presença ou ausência de uma característica específica.

### 2. Count-based methods (sparse):
   - **Explicação:** Estes métodos baseiam-se na contagem de ocorrências de palavras em documentos. A ideia é construir uma matriz que representa a frequência com que cada palavra ocorre em relação a outras palavras.
   - **Exemplos:**
      - **Point-wise Mutual Information (PMI):** Mede a associação entre duas palavras, levando em consideração sua co-ocorrência.
      - **TF-IDF (Term Frequency-Inverse Document Frequency):** Atribui pesos às palavras com base em sua frequência em um documento específico em relação à frequência em todos os documentos.

### 3. Classical dimensionality reduction (dense):
   - **Explicação:** Estes métodos buscam reduzir a dimensionalidade dos dados, preservando as características mais importantes. Isso é feito através de técnicas como a Análise de Componentes Principais (PCA), Singular Value Decomposition (SVD), Latent Semantic Analysis (LSA) e Latent Dirichlet Allocation (LDA).
   - **Exemplos:**
      - **PCA (Principal Component Analysis):** Reduz a dimensionalidade dos dados, mantendo as direções principais de variabilidade.
      - **SVD (Singular Value Decomposition):** Decomposição de uma matriz em três matrizes, permitindo redução de dimensionalidade.
      - **LDA (Latent Dirichlet Allocation):** Modelagem estatística que atribui tópicos a documentos e palavras a tópicos.

### 4. Learned dimensionality (dense):
   - **Explicação:** Esses métodos envolvem a aprendizagem de representações densas de palavras, onde cada palavra é representada por um vetor contínuo de números reais.
   - **Exemplos:**
      - **Autoencoders:** Redes neurais que tentam reconstruir a entrada, aprendendo representações latentes no processo.
      - **Word2Vec:** A técnica que aprende representações distribuídas de palavras usando redes neurais.
      - **GloVe (Global Vectors for Word Representation):** Uma técnica que utiliza estatísticas globais de co-ocorrência para aprender representações de palavras.

Esses métodos descrevem diferentes abordagens para representar palavras como vetores, seja através de características esparsas ou de representações densas, sendo aplicados em diversas tarefas de processamento de linguagem natural.

## Word representations and Context
As representações estáticas possuem o __lack__ de entendimento contextual, já que palavras podem ter significados semânticos diferentes em diferentes contextos.

Quando falamos de representações estáticas de palavras, referimo-nos a técnicas que atribuem a uma palavra um único vetor, independente do contexto em que ela aparece. Isso pode ser problemático porque muitas palavras têm significados polissêmicos, ou seja, podem ter interpretações diferentes dependendo do contexto em que são usadas.

Por exemplo, a palavra "banco" pode se referir a um banco de parque ou a uma instituição financeira. Se utilizarmos uma representação estática, teríamos um único vetor para "banco", ignorando a variação de significado.

A falta de consideração do contexto pode levar a uma perda de nuances semânticas e prejudicar o desempenho em tarefas que exigem compreensão contextual, como análise de sentimentos, tradução automática ou até mesmo resolução de ambiguidades.

Para superar essa limitação, muitas abordagens mais recentes no campo de Processamento de Linguagem Natural (NLP) se concentraram em representações de palavras contextualizadas. Modelos como o BERT (Bidirectional Encoder Representations from Transformers) e o GPT (Generative Pre-trained Transformer) utilizam arquiteturas de transformers para capturar o contexto em que cada palavra aparece, gerando representações mais ricas e sensíveis ao contexto. Isso permite que os modelos compreendam melhor as nuances semânticas e melhorem o desempenho em uma variedade de tarefas linguísticas.

## Model structure and linguistic structure

A estrutura linguística de frases e a estrutura de modelos referem-se a dois conceitos diferentes, mas podemos explorar ambos.

### Estrutura Linguística de Frases:

#### Estrutura Básica:
1. **Sujeito:** Geralmente, a entidade que realiza ou é afetada pela ação na frase.
   - Exemplo: "O gato"
2. **Verbo:** A ação ou estado que a frase expressa.
   - Exemplo: "corre"
3. **Objeto:** A entidade ou coisa que recebe a ação.
   - Exemplo: "no jardim."

#### Estrutura de Exemplo:
"O gato corre no jardim."

### Estrutura de Modelos (como em Modelos de Linguagem):

#### Estrutura de Rede Neural (Exemplo com BERT):
1. **Camada de Entrada:** Recebe a sequência de tokens (palavras ou subpalavras).
2. **Camadas Intermediárias:** Processam a informação, capturando relações semânticas e contextuais.
3. **Camada de Saída:** Gera as representações contextualizadas ou realiza tarefas específicas, como classificação de texto.

#### Estrutura de Exemplo:
- **Entrada:** "O gato corre no jardim."
- **Processamento:** (Camadas intermediárias processam informações contextuais)
- **Saída:** Representações contextuais ou resposta para uma tarefa específica.

### Comparação:

1. **Natureza:**
   - **Estrutura Linguística de Frases:** Descreve como as palavras e componentes de uma frase se organizam para formar significado.
   - **Estrutura de Modelos:** Refere-se à arquitetura e organização de uma rede neural ou modelo de linguagem.

2. **Flexibilidade e Contexto:**
   - **Estrutura Linguística de Frases:** A estrutura é estática e não leva em consideração o contexto mais amplo.
   - **Estrutura de Modelos:** Captura informações contextuais, permitindo uma compreensão mais dinâmica e adaptativa.

3. **Aplicação:**
   - **Estrutura Linguística de Frases:** Aplica-se à análise gramatical e semântica de frases na língua natural.
   - **Estrutura de Modelos:** Aplica-se a modelos de linguagem, que podem realizar várias tarefas, como tradução automática, geração de texto e compreensão de linguagem natural.

4. **Exemplos:**
   - **Estrutura Linguística de Frases:** "O gato corre no jardim."
   - **Estrutura de Modelos:** Entrada e saída de um modelo de linguagem como BERT.

Em resumo, a estrutura linguística de frases lida com a organização de palavras para formar uma unidade significativa, enquanto a estrutura de modelos refere-se à arquitetura e organização de redes neurais ou modelos de linguagem para processar e entender a linguagem natural de maneira mais dinâmica e contextual.

Vamos explorar as estruturas dos modelos GloVe, RNN (Redes Neurais Recorrentes), Tree Structure Networks RNN e Bidirectional RNN.

### 1. GloVe (Global Vectors for Word Representation):

#### Estrutura:
- **Tipo:** Modelo de aprendizado não supervisionado para representações de palavras.
- **Arquitetura:** Não é uma rede neural, mas uma técnica de vetorização de palavras baseada em estatísticas globais de co-ocorrência.
- **Treinamento:** Utiliza uma matriz de co-ocorrência e técnicas de fatoração para aprender representações distribuídas de palavras.

### 2. RNN (Redes Neurais Recorrentes):

#### Estrutura:
- **Tipo:** Modelo de aprendizado supervisionado, usado principalmente para sequências.
- **Arquitetura:** Possui células recorrentes que permitem a informação ser persistente ao longo do tempo. Cada unidade recebe uma entrada e a informação da unidade anterior.
- **Treinamento:** Usa backpropagation através do tempo (BPTT) para ajustar os pesos durante o treinamento.

### 3. Tree Structure Networks RNN:
#### Estrutura:
- **Tipo:** Modelo de aprendizado supervisionado, projetado para processar estruturas de árvores.
- **Arquitetura:** Utiliza estruturas de árvore para modelar relações hierárquicas entre elementos de uma sequência.
- **Treinamento:** Semelhante a uma RNN tradicional, mas com considerações especiais para a estrutura de árvore.

### 4. Bidirectional RNN (RNN Bidirecional):

#### Estrutura:
- **Tipo:** Extensão da RNN padrão.
- **Arquitetura:** Possui duas camadas de células recorrentes, uma processando a sequência da esquerda para a direita e a outra da direita para a esquerda.
- **Vantagem:** Captura informações contextuais de ambos os lados da sequência, melhorando a compreensão contextual.

Essas estruturas representam diferentes abordagens para lidar com dados sequenciais, como texto. O GloVe é uma técnica baseada em estatísticas, enquanto RNN, Tree Structure Networks RNN e Bidirectional RNN são modelos de aprendizado de máquina que incorporam arquiteturas específicas para lidar com a natureza sequencial ou hierárquica dos dados linguísticos. Cada um tem suas aplicações específicas em tarefas de processamento de linguagem natural (NLP) e pode ser escolhido com base nas características do problema em questão.

### 5. GloVe (Global Vectors for Word Representation):

#### Estrutura:
- **Tipo:** Modelo de aprendizado não supervisionado para representações de palavras.
- **Arquitetura:** Não é uma rede neural, mas uma técnica de vetorização de palavras baseada em estatísticas globais de co-ocorrência.
- **Treinamento:** Utiliza uma matriz de co-ocorrência e técnicas de fatoração para aprender representações distribuídas de palavras.

### 6. RNN (Redes Neurais Recorrentes):

#### Estrutura:
- **Tipo:** Modelo de aprendizado supervisionado, usado principalmente para sequências.
- **Arquitetura:** Possui células recorrentes que permitem a informação ser persistente ao longo do tempo. Cada unidade recebe uma entrada e a informação da unidade anterior.
- **Treinamento:** Usa backpropagation através do tempo (BPTT) para ajustar os pesos durante o treinamento.

### 7. Tree Structure Networks RNN:
#### Estrutura:
- **Tipo:** Modelo de aprendizado supervisionado, projetado para processar estruturas de árvores.
- **Arquitetura:** Utiliza estruturas de árvore para modelar relações hierárquicas entre elementos de uma sequência.
- **Treinamento:** Semelhante a uma RNN tradicional, mas com considerações especiais para a estrutura de árvore.

### 8. Bidirectional RNN (RNN Bidirecional):
#### Estrutura:
- **Tipo:** Extensão da RNN padrão.
- **Arquitetura:** Possui duas camadas de células recorrentes, uma processando a sequência da esquerda para a direita e a outra da direita para a esquerda.
- **Vantagem:** Captura informações contextuais de ambos os lados da sequência, melhorando a compreensão contextual.

Essas estruturas representam diferentes abordagens para lidar com dados sequenciais, como texto. O GloVe é uma técnica baseada em estatísticas, enquanto RNN, Tree Structure Networks RNN e Bidirectional RNN são modelos de aprendizado de máquina que incorporam arquiteturas específicas para lidar com a natureza sequencial ou hierárquica dos dados linguísticos. Cada um tem suas aplicações específicas em tarefas de processamento de linguagem natural (NLP) e pode ser escolhido com base nas características do problema em questão.

## Attention
O mecanismo de atenção é uma técnica fundamental em modelos de aprendizado de máquina, especialmente em tarefas de processamento de linguagem natural (NLP), onde é crucial considerar diferentes partes de uma sequência de entrada de maneira ponderada. O mecanismo de atenção permite que o modelo atribua diferentes pesos a diferentes partes da entrada, destacando as informações mais relevantes para a tarefa em questão.

### Mecanismo de Atenção em NLP:

1. **Entrada Sequencial:**
   - Suponha que temos uma sequência de palavras ou vetores de entrada, como uma frase em um problema de NLP.

2. **Representação de Consulta (Query):**
   - Uma representação da "consulta" é criada. Pode ser um vetor associado a uma palavra ou a uma representação mais abstrata do contexto.

3. **Pontuação de Atenção (Attention Score):**
   - Calcula-se uma pontuação de atenção para cada elemento da sequência em relação à consulta. Essa pontuação reflete a importância relativa de cada elemento.
   - A pontuação de atenção é geralmente calculada usando funções de similaridade, como o produto escalar ou uma função de similaridade mais complexa.

4. **Pesos Normalizados:**
   - As pontuações de atenção são normalizadas usando uma função softmax para obter pesos que somam 1. Esses pesos indicam a importância relativa de cada elemento na sequência.

5. **Atenção Ponderada:**
   - Os elementos da sequência são ponderados pelos pesos obtidos. Isso cria uma representação ponderada que destaca partes específicas da entrada, com base na consulta.

### Exemplo Simples:

Considere a frase: "O gato está dormindo no sofá."

- **Consulta:** Pode ser uma representação contextual, como um vetor que representa o significado de uma palavra específica.
- **Pontuação de Atenção:** Calcula a similaridade entre a consulta e cada palavra da frase.
- **Pesos Normalizados:** Aplica a função softmax nas pontuações para obter pesos normalizados.
- **Atenção Ponderada:** Combina as palavras ponderadas pela atenção para obter uma representação contextualizada.

### Aplicações:

- **Tradução Automática:** Destacar palavras relevantes em uma frase de origem para a tradução.
- **Sumarização de Texto:** Identificar partes importantes de um texto para criar uma sumarização concisa.
- **Processamento de Perguntas e Respostas:** Focar em partes relevantes do contexto para gerar respostas.

Modelos como o Transformer, que introduziu a atenção multi-cabeça, foram revolucionários em NLP, proporcionando uma capacidade aprimorada de lidar com sequências de maneira mais sofisticada.

## Subword Modeling in ELMo
A modelagem de subpalavras no ELMo (Embeddings from Language Models) é uma abordagem que utiliza representações contextuais para palavras, levando em consideração subpalavras ou morfemas, em vez de considerar as palavras como unidades inteiras. Essa técnica visa capturar a riqueza e a flexibilidade da linguagem ao lidar com a variação morfológica e a composição de palavras.

A modelagem de subpalavras no ELMo é alcançada por meio de uma arquitetura de rede neural profunda. Aqui está uma explicação mais detalhada:

### Estrutura Geral do ELMo:

1. **Rede Neural Bidirecional (Bi-LSTM):**
   - Utiliza uma arquitetura de LSTM bidirecional para processar a sequência de entrada.
   - A LSTM bidirecional permite que a rede capture informações contextuais de ambas as direções, à esquerda e à direita da palavra.

2. **Camadas de Projeção Linear:**
   - Após a camada Bi-LSTM, cada camada oculta é projetada linearmente para reduzir a dimensionalidade e extrair representações mais ricas.

3. **Composição de Camadas:**
   - A representação de cada palavra é composta por diferentes camadas da rede. Cada camada captura diferentes níveis de complexidade e contexto.

4. **Combinação Ponderada de Representações:**
   - As representações de cada camada são combinadas ponderadamente para formar a representação final da palavra. Essa combinação é feita usando pesos aprendidos durante o treinamento.

### Modelagem de Subpalavras:

- **Subpalavras como Unidades:**
  - Ao contrário de abordagens que tratam palavras como unidades discretas, o ELMo considera as subpalavras ou morfemas como unidades fundamentais.
  - Isso permite que o modelo capture informações sobre a morfologia e a composição das palavras.

- **Embeddings Contextuais:**
  - As representações de subpalavras são contextuais, ou seja, dependem do contexto em que a subpalavra aparece na sequência.
  - Essa contextualização é fundamental para lidar com a polissemia e a composição de palavras em diferentes contextos.

### Vantagens da Modelagem de Subpalavras no ELMo:

- **Flexibilidade Morfológica:** Permite ao modelo lidar com diferentes formas morfológicas de uma palavra.
- **Composicionalidade:** Captura a composição semântica de palavras complexas ou raras.
- **Polissemia:** Lida melhor com palavras que têm significados variados em contextos diferentes.

A modelagem de subpalavras no ELMo contribui para a criação de embeddings mais ricos e contextuais, resultando em representações mais sofisticadas para palavras em tarefas de processamento de linguagem natural.

## Positional Encoding
O positional encoding (codificação posicional) é uma técnica usada em modelos de linguagem, especialmente em arquiteturas como o Transformer, para incorporar informações sobre a posição relativa das palavras em uma sequência. Essa técnica é necessária porque modelos de linguagem, como redes neurais, por si só, não têm uma noção intrínseca de ordem ou posição nas sequências de entrada.

Nos modelos que utilizam codificação posicional, é adicionado um vetor posicional a cada vetor de entrada, representando a posição relativa da palavra na sequência. Dessa forma, a rede neural é capacitada para considerar a posição das palavras, algo essencial em tarefas onde a ordem das palavras é significativa, como em tradução de texto ou análise de sentimento em sequências.

### Formulação Básica do Positional Encoding:

Seja \( PE(pos, 2i) \) o \( 2i \)-ésimo componente do vetor de codificação posicional na posição \( pos \), e \( PE(pos, 2i + 1) \) o \( (2i + 1) \)-ésimo componente. A fórmula básica do positional encoding é geralmente definida como:

\[ PE(pos, 2i) = \sin\left(\frac{{pos}}{{10000^{(2i/d)}}}\right) \]

\[ PE(pos, 2i + 1) = \cos\left(\frac{{pos}}{{10000^{(2i/d)}}}\right) \]

onde:
- \( pos \) é a posição da palavra na sequência.
- \( i \) é a dimensão do vetor de codificação posicional.
- \( d \) é a dimensão total do vetor de entrada.

### Propriedades Importantes:

1. **Ponderação por Dimensão:**
   - A ponderação na fórmula garante que diferentes dimensões do vetor de codificação posicional capturam padrões em diferentes escalas de posição.

2. **Sinusoides Alternadas:**
   - A escolha de funções senoidais e cossenoidais alterna entre as dimensões para que elas capturem padrões diferentes e evitem a perda de informação.

3. **Adição ao Vetor de Entrada:**
   - O positional encoding é somado ao vetor de entrada original, permitindo que a informação de posição seja incorporada à representação da palavra.

O positional encoding é uma abordagem eficaz para lidar com a informação de posição em modelos que não possuem uma compreensão intrínseca da ordem nas sequências, contribuindo para o desempenho em tarefas que requerem consideração da posição relativa das palavras.

## Fine-tuning
O fine-tuning em Processamento de Linguagem Natural (NLP) é um processo em que um modelo de linguagem pré-treinado, geralmente treinado em uma tarefa de linguagem geral, é ajustado para realizar uma tarefa específica relacionada. Isso é particularmente útil quando se dispõe de dados limitados para a tarefa específica, pois o modelo pré-treinado já adquiriu conhecimento linguístico geral de um conjunto diversificado e extenso de dados.

Aqui está uma explicação mais detalhada sobre como o fine-tuning é realizado em tarefas de NLP:

### Processo de Fine-Tuning em NLP:

1. **Escolha do Modelo Pré-Treinado:**
   - Seleção de um modelo pré-treinado bem estabelecido, treinado em grandes quantidades de dados em uma tarefa geral de NLP. Exemplos incluem modelos como BERT, GPT, ELMo, entre outros.

2. **Congelamento e Ajuste de Camadas:**
   - Congelamento das camadas iniciais do modelo pré-treinado, que capturam conhecimentos linguísticos gerais, para preservar essas informações.
   - Ajuste das camadas finais, mais próximas da saída, para a nova tarefa específica.

3. **Adição de Camadas Específicas:**
   - Adição de camadas específicas para a nova tarefa. Por exemplo, na classificação de sentimentos, pode-se adicionar uma camada de classificação no topo do modelo.

4. **Ajuste da Taxa de Aprendizado:**
   - Redução da taxa de aprendizado para as camadas ajustadas durante o fine-tuning. Isso ajuda a realizar ajustes mais graduais e evitar grandes mudanças nas representações já aprendidas.

5. **Treinamento na Tarefa Específica:**
   - Treinamento do modelo no conjunto de dados específico da nova tarefa, utilizando a combinação de camadas congeladas e camadas ajustadas.

6. **Ajustes no Conjunto de Dados:**
   - É possível ajustar o conjunto de dados da nova tarefa, adicionando ou removendo exemplos relevantes para aprimorar o desempenho do fine-tuning.

### Exemplos de Tarefas em NLP que Usam Fine-Tuning:

1. **Classificação de Sentimentos:**
   - Fine-tuning de modelos para classificar se um texto expressa um sentimento positivo, negativo ou neutro.

2. **Identificação de Entidades Nomeadas (NER):**
   - Adaptação de modelos para reconhecer entidades específicas, como nomes de pessoas, organizações ou locais, em textos.

3. **Preenchimento de Lacunas (Cloze Test):**
   - Ajuste de modelos para preencher lacunas em frases ou parágrafos, onde uma palavra ou trecho é removido.

4. **Tradução Automática:**
   - Fine-tuning de modelos para tarefas de tradução automática, adaptando modelos de linguagem para pares de idiomas específicos.

O fine-tuning em NLP é uma estratégia poderosa para adaptar modelos pré-treinados a tarefas específicas de linguagem, permitindo uma utilização mais eficiente de recursos computacionais e dados.
