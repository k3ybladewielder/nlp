# Stanford XCS224U: Natural Language Understanding
- O XCS224u tem o curso CS224n como prerequisito.[Background Materials](https://web.stanford.edu/class/cs224u/background.html)
- [Materiais do curso](https://web.stanford.edu/class/cs224u/index.html)

# Summary
> INSERIR SUMÁRIO

> INSERIR TOPICOS PARA SUMARIO SEGUINDO A EMENTA

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

A estrutura linguística de frases e a estrutura de modelos referem-se a dois conceitos diferentes.

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

# Tranformers Core Model Structure
A arquitetura dos modelos Transformers, introduzida por Vaswani et al. em 2017, é uma arquitetura de aprendizado de máquina que se tornou amplamente utilizada em tarefas de Processamento de Linguagem Natural (NLP) e além. A característica distintiva dos modelos Transformers é a atenção, que permite que o modelo atenda simultaneamente a todas as posições de entrada em uma sequência, em vez de depender de uma abordagem sequencial ou recorrente.

Aqui estão os principais componentes da arquitetura Transformer:

1. **Encoder-Decoder Architecture:**
   - Muitos modelos Transformer consistem em uma arquitetura de codificador-decodificador. No caso de tarefas de tradução, por exemplo, o codificador processa a sequência de entrada na língua de origem, e o decodificador gera a sequência de saída na língua de destino.

2. **Attention Mechanism:**
   - O mecanismo de atenção é central para os Transformers. Ele permite que o modelo "preste atenção" a diferentes partes da entrada ao calcular uma pontuação de atenção para cada posição. Isso é feito por meio de três vetores: consulta (Q), chave (K) e valor (V). A atenção é calculada como uma soma ponderada dos valores, onde os pesos são determinados pela compatibilidade entre a consulta e a chave.
   - Exemplo: Considere a frase "The cat sat on the mat." O mecanismo de atenção permite que o modelo dê mais peso a diferentes palavras dependendo do contexto. Por exemplo, ao traduzir para outra língua, o modelo pode dar mais atenção a "cat" ao gerar a palavra correspondente na língua de destino.

3. **Multi-Head Attention:**
   - Para melhorar a representação, os modelos Transformers usam atenção multi-cabeça, onde várias cabeças de atenção são calculadas independentemente e, em seguida, concatenadas e linearmente transformadas. Isso permite que o modelo aprenda diferentes representações ponderadas.
   - Exemplo: Suponha que estamos analisando uma frase e queremos entender tanto os sujeitos quanto os objetos. Cada cabeça de atenção pode se concentrar em diferentes partes da frase, como sujeitos ou objetos específicos.

4. **Positional Encoding:**
   - Uma desvantagem dos modelos Transformers é que eles não têm uma noção intrínseca de ordem ou posição nas sequências. Para contornar isso, os modelos incorporam informações de posição através da adição de codificações posicionais às embeddings de entrada.
   - Exemplo: Considere as frases "I love transformers" e "Transformers love me." As palavras têm significados diferentes dependendo de sua posição nas frases. As codificações posicionais ajudam o modelo a capturar essas diferenças.

5. **Feedforward Neural Networks:**
   - Após a camada de atenção, cada posição passa por uma rede neural feedforward, adicionando uma camada de não-linearidade. Isso ajuda o modelo a aprender representações não lineares mais complexas, capturando relações mais abstratas entre as palavras.

6. **Layer Normalization and Residual Connections:**
   - Cada subcamada (atualmente, subcamadas são atenção e feedforward) é seguida por uma camada de normalização e uma conexão residual. Isso ajuda na estabilidade do treinamento e facilita o fluxo de gradientes.
   - Exemplo: A normalização de camada e as conexões residuais são usadas para melhorar a estabilidade do treinamento e facilitar o fluxo de gradientes. Isso é particularmente útil em modelos profundos como os Transformers.

7. **Self-Attention:**
   - Em muitos casos, como no BERT, os modelos Transformers usam autoatenção, onde a entrada é considerada para calcular as atenções em si mesma. Isso permite que o modelo capture dependências de longo alcance.
   - Exemplo: Ao processar uma sequência de palavras, o mecanismo de autoatenção permite que cada palavra considere todas as outras palavras em relação a si mesma. Isso é valioso para entender dependências de longo alcance.
   
# BERT Core Model Components
O BERT (Bidirectional Encoder Representations from Transformers) é um modelo de linguagem pré-treinado baseado na arquitetura Transformer. Aqui estão os componentes essenciais do BERT:

1. **Arquitetura Transformer:**
   - BERT utiliza a arquitetura Transformer, composta por um codificador empregado em uma configuração bidirecional. Isso significa que o modelo leva em consideração as palavras anteriores e posteriores para cada palavra em uma frase durante o treinamento.

2. **Camadas de Ativação e Normalização:**
   - BERT incorpora camadas de ativação (como ReLU) e normalização de camada após as operações de atenção e redes neurais feedforward. Essas camadas contribuem para a estabilidade do treinamento e facilitam o fluxo de gradientes.

3. **Multi-Head Self-Attention:**
   - O mecanismo de autoatenção é aplicado em várias cabeças (multi-head attention), permitindo que o modelo capture diferentes aspectos de dependências em uma frase de maneira simultânea. Isso ajuda BERT a entender contextos complexos e relações entre palavras.

4. **Embeddings Posicionais:**
   - Dado que a arquitetura Transformer não leva em conta a ordem das palavras em uma frase, BERT incorpora informações de posição através de embeddings posicionais. Isso permite que o modelo diferencie entre palavras que ocorrem em diferentes posições dentro de uma sequência.

5. **Camada de Token [CLS]:**
   - BERT adiciona um token especial [CLS] (CLS token) no início de cada sequência de entrada. O vetor de representação associado ao token [CLS] é usado como uma representação agregada da frase inteira para tarefas de classificação.

6. **Pré-treinamento com Masked Language Model (MLM):**
   - Durante a fase de pré-treinamento, BERT treina um modelo de linguagem de forma bidirecional. Uma parte das palavras em cada sequência de entrada é mascarada, e o modelo é treinado para prever essas palavras mascaradas com base no contexto das palavras circundantes.

7. **Pré-treinamento de Palavras Inteiras (Whole Word Masking):**
   - BERT usa uma abordagem de "Whole Word Masking" durante o pré-treinamento, na qual palavras inteiras são mascaradas de uma vez. Isso ajuda o modelo a entender o contexto e a relação semântica entre palavras completas.

8. **Fine-Tuning para Tarefas Específicas:**
   - Após o pré-treinamento, BERT pode ser afinado para tarefas específicas, como classificação de sentimentos, perguntas e respostas, ou NER (Reconhecimento de Entidade Nomeada), utilizando camadas adicionais e ajustando parâmetros para a tarefa específica.

Esses componentes tornam o BERT uma arquitetura poderosa e versátil para uma variedade de tarefas de Processamento de Linguagem Natural (NLP), permitindo que o modelo compreenda contextos complexos e relações semânticas em sequências de texto.

# RoBERTa Core Model Components
RoBERTa (Robustly optimized BERT approach with pre-training Larger Amount of data) é uma variação do modelo BERT projetada para otimizar o desempenho e a eficiência do treinamento. Aqui estão os componentes centrais do ROBERTA:

1. **Arquitetura Baseada em Transformer:** ROBERTA mantém a arquitetura baseada em Transformer, herdada do BERT. A arquitetura é composta por camadas de autoatenção, camadas de feedforward, normalização de camada e conexões residuais.
2. **Treinamento Escalonado e Batch Size Dinâmico:** ROBERTA utiliza uma abordagem de treinamento escalonado (layer-wise training) e ajusta o tamanho do lote dinamicamente durante o treinamento. Isso permite que camadas mais profundas se beneficiem de tamanhos de lote maiores, o que melhora o desempenho do modelo.
3. **Remoção da Pré-treinamento de Sentença (NSP):** ROBERTA remove a tarefa de pré-treinamento de sentença (Next Sentence Prediction - NSP) usada no BERT. Em vez disso, ele pré-treina o modelo apenas com tarefas de preenchimento de máscara (Masked Language Model - MLM) e utiliza um conjunto de dados mais extenso.
4. **Tokenização Dinâmica e Aprendizado Contínuo:** ROBERTA incorpora uma abordagem de tokenização dinâmica, o que significa que o tamanho do vocabulário pode ser expandido dinamicamente durante o treinamento. Isso é especialmente útil para lidar com grandes quantidades de dados.
5. **Aumento do Tamanho do Modelo e Treinamento com Mais Dados:** ROBERTA aumenta o tamanho do modelo em comparação com o BERT padrão e é treinado com uma quantidade significativamente maior de dados. Isso ajuda a capturar uma representação mais rica e robusta das linguagens.
6. **Atenção Contínua em Segmentos (Causal Language Modeling):** Durante o treinamento, ROBERTA introduz a atenção contínua em segmentos para lidar com tokens de segmentos em um fluxo contínuo, melhorando a capacidade do modelo de entender e representar contextos complexos.
7. **Utilização de Stop Words no Pré-treinamento:** ROBERTA faz uso de palavras comuns (stop words) no pré-treinamento, o que pode ajudar a melhorar a capacidade do modelo de compreender e generalizar.
8. **Redução de Memória e Processamento Eficiente:** ROBERTA utiliza estratégias eficientes para redução de memória, como compartilhamento de parâmetros e compressão de representações intermediárias, tornando-o mais escalável e eficiente em termos de recursos computacionais.

Esses componentes fazem do ROBERTA uma extensão e otimização do BERT, resultando em um modelo mais robusto e eficiente para tarefas de Processamento de Linguagem Natural.

# BERT vs RoBERTa
1. **Treinamento Escalonado (Layer-wise Training):**
   - **BERT:** Treina todas as camadas simultaneamente.
   - **ROBERTA:** Utiliza treinamento escalonado, treinando camadas mais profundas com tamanhos de lote maiores.

2. **Tamanho do Lote Dinâmico:**
   - **BERT:** Usa um tamanho de lote constante durante o treinamento.
   - **ROBERTA:** Adapta dinamicamente o tamanho do lote, aumentando-o para camadas mais profundas.

3. **Pré-treinamento de Sentença (NSP):**
   - **BERT:** Inclui a tarefa NSP (Next Sentence Prediction) no pré-treinamento.
   - **ROBERTA:** Remove a tarefa NSP e foca exclusivamente em MLM (Masked Language Model).

4. **Tamanho do Modelo:**
   - **BERT:** Tamanho de modelo padrão.
   - **ROBERTA:** Aumenta o tamanho do modelo, proporcionando maior capacidade de representação.

5. **Tokenização Dinâmica:**
   - **BERT:** Usa um vocabulário estático durante todo o treinamento.
   - **ROBERTA:** Incorpora tokenização dinâmica, permitindo expansão do vocabulário durante o treinamento.

6. **Quantidade de Dados de Treinamento:**
   - **BERT:** Treinado com um conjunto de dados específico.
   - **ROBERTA:** Treinado com uma quantidade significativamente maior de dados.

7. **Atenção Contínua em Segmentos (Causal Language Modeling):**
   - **BERT:** Não inclui atenção contínua em segmentos.
   - **ROBERTA:** Introduz atenção contínua em segmentos para lidar com tokens de segmentos em um fluxo contínuo.

8. **Utilização de Stop Words no Pré-treinamento:**
   - **BERT:** Não especificamente projetado para incluir stop words no pré-treinamento.
   - **ROBERTA:** Faz uso de stop words no pré-treinamento para melhorar a compreensão e generalização.

9. **Redução de Memória e Processamento Eficiente:**
   - **BERT:** Não implementa estratégias específicas de redução de memória.
   - **ROBERTA:** Usa estratégias eficientes, como compartilhamento de parâmetros e compressão de representações intermediárias.

Em resumo, o ROBERTA é uma extensão otimizada do BERT, incorporando estratégias como treinamento escalonado, aumento do tamanho do modelo, tokenização dinâmica e utilização de stop words no pré-treinamento para melhorar o desempenho e a eficiência do modelo. Essas modificações tornam o ROBERTA mais robusto e eficiente em relação a determinados aspectos do treinamento e da representação linguística.

# ELECTRA Core Model Components
O ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately) é um modelo de linguagem pré-treinado que se destaca por sua eficiência de treinamento e bom desempenho em tarefas downstream*. Aqui estão os principais componentes do ELECTRA:

1. **Generador de Tokens Masked (Geração de Token Mascaramento):**  Diferentemente do BERT, que usa uma abordagem de preenchimento de máscara (MLM) para mascarar aleatoriamente palavras em uma sequência, o ELECTRA usa um gerador de tokens mascarados. Este gerador substitui aleatoriamente palavras reais por [MASK] e treina o modelo para reconhecer essas substituições.
2. **Discriminador de Tokens Substituídos:**  O ELECTRA introduz um discriminador que é treinado para distinguir tokens reais de tokens gerados pelo gerador de tokens mascarados. Esse componente é crucial para a abordagem de treinamento do ELECTRA.
3. **Tarefas de Pré-treinamento:**  O modelo é pré-treinado em duas tarefas simultâneas: a tarefa de reconhecimento de tokens mascarados (MLM-like) e a tarefa de discriminação adversarial entre tokens reais e tokens gerados.
4. **Substituição de Tokens por [MASK]:**  No ELECTRA, uma porcentagem significativa dos tokens é substituída por [MASK] durante o pré-treinamento, em comparação com a pequena fração de tokens mascarados no BERT. Isso cria um sinal de treinamento mais forte para o modelo.
5. **Treinamento Adversarial:**  O treinamento adversarial entre o gerador e o discriminador é uma parte central do ELECTRA. O gerador tenta gerar tokens mascarados que se assemelham a tokens reais, enquanto o discriminador tenta distinguir entre tokens reais e gerados.
6. **Estratégia de Substituição Dinâmica:** O ELECTRA utiliza uma estratégia de substituição dinâmica, onde alguns tokens são substituídos por [MASK], alguns permanecem inalterados e outros são substituídos por palavras reais. Essa abordagem permite um treinamento mais eficiente.
7. **Aproveitamento de Modelos de Linguagem Pré-existentes:** O ELECTRA pode se beneficiar de modelos de linguagem pré-existentes, como o BERT, para inicialização de parâmetros antes do treinamento adversarial.
8. **Desempenho Eficiente em Tarefas Downstream:** Devido à sua abordagem de treinamento eficiente e ao uso de uma quantidade significativamente menor de parâmetros em comparação com modelos tradicionais, o ELECTRA demonstrou bom desempenho em tarefas downstream com menos recursos computacionais."Tarefas downstream" referem-se a tarefas específicas de Processamento de Linguagem Natural (PLN) que são resolvidas utilizando modelos de linguagem pré-treinados. Em um contexto de modelos de linguagem pré-treinados, o termo "downstream" refere-se ao movimento de uma tarefa mais geral (pré-treinamento) para tarefas mais específicas e aplicadas (tarefas downstream). Ex: 
 - Classificação de Sentimento, Perguntas e Respostas (QA), Named Entity Recognition (NER), Tradução Automátic Geração de Texto, Sumarização de Texto, Análise de Sentimento.

Em resumo, o ELECTRA se destaca pela sua eficiência de treinamento, aproveitando um gerador de tokens mascarados e um discriminador adversarial para pré-treinamento. Essa abordagem adversarial resulta em representações mais ricas e eficazes para várias tarefas de Processamento de Linguagem Natural (NLP).

# Bert vs RoBERta e ELECTRA
Vamos comparar as principais diferenças entre o BERT, RoBERTa e ELECTRA:

1. **Arquitetura Base:**
   - **BERT:** Utiliza a arquitetura Transformer com atenção bidirecional.
   - **RoBERTa:** Também utiliza a arquitetura Transformer, com algumas otimizações, como treinamento escalonado e tamanho de lote dinâmico.
   - **ELECTRA:** Introduz uma abordagem adversarial, com um gerador de tokens mascarados e um discriminador adversarial.

2. **Tarefas de Pré-treinamento:**
   - **BERT:** Pré-treina o modelo com tarefas de preenchimento de máscara (MLM) e previsão de sentença seguinte (Next Sentence Prediction - NSP).
   - **RoBERTa:** Foca principalmente na tarefa de preenchimento de máscara (MLM), removendo a tarefa NSP.
   - **ELECTRA:** Introduz uma tarefa adversarial, substituindo tokens reais por tokens mascarados e treinando um discriminador para distinguir entre tokens reais e gerados.

3. **Preenchimento de Máscara (MLM):**
   - **BERT:** Usa uma estratégia de preenchimento de máscara padrão.
   - **RoBERTa:** Aprimora a estratégia de preenchimento de máscara com uma porcentagem maior de tokens mascarados e tamanhos de lote dinâmicos.
   - **ELECTRA:** Substitui aleatoriamente palavras reais por tokens mascarados durante o pré-treinamento adversarial.

4. **Treinamento Adversarial:**
   - **BERT:** Não incorpora um treinamento adversarial explícito.
   - **RoBERTa:** Não introduz treinamento adversarial, mas otimizações como treinamento escalonado.
   - **ELECTRA:** Treina o modelo adversarialmente com um gerador de tokens mascarados e um discriminador adversarial.

5. **Aumento do Tamanho do Modelo:**
   - **BERT:** Tamanho de modelo padrão.
   - **RoBERTa:** Aumenta o tamanho do modelo para melhorar a capacidade de representação.
   - **ELECTRA:** Usa uma quantidade significativamente menor de parâmetros comparado a modelos tradicionais.

6. **Desempenho em Tarefas Downstream:**
   - **BERT:** Demonstra bom desempenho em tarefas downstream.
   - **RoBERTa:** Tende a superar o desempenho do BERT em várias tarefas devido a otimizações e pré-treinamento mais extensivo.
   - **ELECTRA:** Também mostra desempenho competitivo em tarefas downstream, especialmente em cenários com recursos computacionais limitados.

Essas são algumas das diferenças fundamentais entre o BERT, RoBERTa e ELECTRA. Cada um desses modelos apresenta inovações específicas para melhorar a eficiência do pré-treinamento e o desempenho em tarefas downstream.

# Comparativo Geral dos componentes principais (core components) do BERT, RoBERTa e ELECTRA:
### BERT (Bidirectional Encoder Representations from Transformers):

1. **MLM (Masked Language Model):**
   - **Descrição:** Usa a tarefa de preenchimento de máscara, onde uma pequena porcentagem de palavras em uma sequência é mascarada e o modelo é treinado para prever essas palavras mascaradas.
   - **Papel:** Enriquecer as representações aprendendo contextos bidirecionais.

2. **NSP (Next Sentence Prediction):**
   - **Descrição:** Tarefa adicional para prever se uma frase segue a outra em um par de sentenças.
   - **Papel:** Capturar relacionamentos entre sentenças em tarefas que exigem compreensão de contexto global.

### RoBERTa (Robustly optimized BERT approach with pre-training Larger Amount of data):

1. **MLM (Masked Language Model):**
   - **Descrição:** Similar ao BERT, mas com uma abordagem mais intensiva, mascarando uma porcentagem maior de palavras e usando tamanhos de lote dinâmicos.
   - **Papel:** Aprimorar a capacidade de representação com treinamento mais extensivo.

2. **NSP (Next Sentence Prediction):**
   - **Descrição:** Removido da arquitetura original do BERT.
   - **Papel:** Eliminar redundâncias e enfocar principalmente na tarefa de preenchimento de máscara.

### ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately):

1. **Generador de Tokens Masked (Geração de Token Mascaramento):**
   - **Descrição:** Substitui aleatoriamente palavras reais por tokens mascarados e treina o modelo adversarialmente para distinguir tokens reais de tokens gerados pelo gerador.
   - **Papel:** Fortalecer o sinal de treinamento e criar representações mais discriminativas.

2. **Discriminador de Tokens Substituídos:**
   - **Descrição:** Treinado para distinguir entre tokens reais e tokens gerados pelo gerador.
   - **Papel:** Avaliar a autenticidade dos tokens durante o treinamento adversarial.

3. **Tarefas de Pré-treinamento:**
   - **Descrição:** Pré-treina o modelo simultaneamente em tarefas de preenchimento de máscara e discriminação adversarial.
   - **Papel:** Aprimorar representações por meio de treinamento adversarial.

### Comparação Geral:

- **Abordagem de Preenchimento de Máscara (MLM):**
  - **BERT:** Preenchimento de máscara padrão.
  - **RoBERTa:** Preenchimento de máscara mais intensivo.
  - **ELECTRA:** Abordagem adversarial com substituição dinâmica.

- **Next Sentence Prediction (NSP):**
  - **BERT:** Incluído originalmente.
  - **RoBERTa:** Removido para focar mais no MLM.
  - **ELECTRA:** Não incorpora explicitamente.

- **Treinamento Adversarial:**
  - **BERT e RoBERTa:** Não introduzem treinamento adversarial.
  - **ELECTRA:** Treinamento adversarial fundamental com gerador e discriminador.
  
# Seq2seq Models
Os modelos sequência para sequência, também conhecidos como seq2seq, são uma classe de modelos de aprendizado de máquina que são usados para tarefas onde a entrada e a saída são sequências de dados. Eles são comumente usados em tarefas de Processamento de Linguagem Natural (PNL) e tradução automática. Aqui está uma explicação geral, uma descrição do funcionamento e exemplos de modelos seq2seq:

1. **Entrada e Saída Sequenciais:** Os modelos seq2seq são projetados para lidar com entradas e saídas que são sequências, como frases em linguagem natural.
2. **Arquitetura Encoder-Decoder:** Geralmente, os modelos seq2seq consistem em duas partes principais: um encoder (codificador) e um decoder (decodificador). O encoder processa a entrada sequencial e produz um vetor de contexto. O decoder usa esse vetor de contexto para gerar a saída sequencial.
3. **Representação Vetorial:** Durante o treinamento, o modelo aprende representações vetoriais que capturam a semântica da sequência de entrada.

### Funcionamento:
1. **Encoder:** A sequência de entrada é alimentada ao encoder, que converte cada elemento da sequência em uma representação vetorial.
2. **Vetor de Contexto:** O encoder produz um vetor de contexto que captura as informações relevantes da sequência de entrada.
3. **Decoder:** O vetor de contexto é usado como entrada para o decoder, que gera a sequência de saída um elemento de cada vez.
4. **Treinamento com Teacher Forcing:** Durante o treinamento, o modelo é alimentado com pares de sequências de entrada e saída conhecidas. O decoder é incentivado a gerar a sequência de saída correta em cada etapa.
5. **Inferência:** Durante a inferência, o modelo é alimentado com uma sequência de entrada e usa o decoder para gerar a sequência de saída.
6. **Exemplos de Modelos Seq2Seq:** Tradução Automática, Geração de Resumo, Diálogos (QA), Geração de Código, Correção de Texto
 
# T5 Core Components 
O T5 (Text-to-Text Transfer Transformer) é um modelo de linguagem proposto pelo Google Research que segue a abordagem "text-to-text", tratando todas as tarefas de processamento de linguagem natural (PNL) como problemas de conversão de texto para texto. Aqui estão os principais componentes do T5:

### 1. **Arquitetura Transformer:**
   - O T5 utiliza a arquitetura Transformer, que é baseada em mecanismos de atenção para capturar relações de longo alcance em sequências.

### 2. **Encoder-Decoder Framework:**
   - Assim como muitos modelos seq2seq, o T5 possui uma estrutura de codificador-decodificador (encoder-decoder). O codificador processa a entrada e gera uma representação contextual, enquanto o decodificador usa essa representação para gerar a saída.

### 3. **Text-to-Text Paradigm:**
   - A característica distintiva do T5 é a abordagem "text-to-text", onde todas as tarefas são formuladas como problemas de conversão de texto para texto. Isso inclui tarefas de classificação, geração, tradução, entre outras.

### 4. **Tokenização Universal:**
   - O T5 utiliza uma tokenização universal, tratando todas as tarefas como tarefas de geração de texto. Cada entrada é prefixada com um rótulo que indica a tarefa específica.

### 5. **Pesos Compartilhados:**
   - O T5 compartilha pesos entre o codificador e o decodificador. Isso contribui para um treinamento eficiente e uma representação mais coerente.

### 6. **Treinamento Multi-Tarefa:**
   - O modelo é treinado simultaneamente em várias tarefas usando um único modelo. Isso permite que o T5 generalize melhor em uma ampla variedade de tarefas de PNL.

### 7. **Transfer Learning:**
   - O T5 é pré-treinado em grandes conjuntos de dados e, em seguida, ajustado finamente para tarefas específicas. Isso segue a abordagem de transfer learning, onde o conhecimento aprendido em uma tarefa é transferido para melhorar o desempenho em outras tarefas.

### Exemplo de Uso do T5:

Suponha que temos uma tarefa de tradução de inglês para francês. A entrada seria algo como "translate English to French: 'The cat is on the mat'" e a saída esperada seria a tradução correspondente para o francês.

1. **Entrada:**
   - "translate English to French: 'The cat is on the mat'"

2. **Saída Esperada:**
   - "Le chat est sur le tapis"

Neste exemplo, a tarefa específica (tradução) é indicada pelo rótulo na entrada, e o modelo é treinado para gerar a saída desejada.

O T5 é conhecido por sua versatilidade e desempenho sólido em uma variedade de tarefas, tornando-o uma escolha popular para abordagens text-to-text em PNL.

# BART Core Components
O BART (Bidirectional and Auto-Regressive Transformers) é um modelo de linguagem proposto pela Facebook AI Research (FAIR) que utiliza a arquitetura Transformer. Ele foi projetado para realizar tarefas de geração de sequência e compressão de texto. Aqui estão os principais componentes do BART:

### 1. **Arquitetura Transformer:**
   - O BART utiliza a arquitetura Transformer, que é composta por camadas de autoatentividade para processar informações de entrada.

### 2. **Encoder-Decoder Framework:**
   - O BART segue uma estrutura de codificador-decodificador (encoder-decoder), onde o codificador processa a entrada e gera uma representação contextual, e o decodificador usa essa representação para gerar a saída.

### 3. **Tokenização:**
   - Assim como outros modelos de linguagem, o BART faz uso de uma estratégia de tokenização para dividir o texto em unidades discretas, como palavras ou subpalavras.

### 4. **BART como Modelo Denoising:**
   - O treinamento do BART é formulado como um problema de denoising autoencoder. Ele é treinado para reconstruir a sequência original a partir de uma versão corrompida da sequência, onde partes aleatórias foram mascaradas ou removidas.

### 5. **Masked Language Model (MLM):**
   - Durante o treinamento, o BART utiliza uma versão modificada da tarefa de preenchimento de máscara (MLM), onde uma parte da sequência é mascarada e o modelo é treinado para prever essas partes mascaradas.

### 6. **Inversão de Sequência no Codificador:**
   - Uma característica única do BART é a inversão da sequência no codificador. Isso significa que a ordem das palavras na entrada é invertida antes de ser passada para o codificador.

### 7. **Fine-Tuning para Tarefas Específicas:**
   - Após o pré-treinamento, o BART pode ser ajustado finamente (fine-tuned) para tarefas específicas, como resumo de texto, tradução automática, entre outras.

### 8. **Geração de Sequências de Saída:**
   - Durante a geração de sequências de saída, o BART é usado para produzir uma sequência de palavras ou subpalavras que representa a resposta desejada.

### Exemplo de Uso do BART:

Suponha que temos a seguinte tarefa de resumo de texto:

1. **Entrada:**
   - "O BART é um modelo de linguagem baseado na arquitetura Transformer. Ele é projetado para realizar tarefas de geração de sequência e compressão de texto."

2. **Saída Esperada:**
   - "O BART, baseado na arquitetura Transformer, é especializado em geração de sequência e compressão de texto."

Neste exemplo, o BART seria treinado para gerar automaticamente o resumo da entrada.

O BART é conhecido por sua eficácia em tarefas de geração de sequência e resumo de texto, e seu treinamento denoising autoencoder contribui para a capacidade do modelo de compreender e gerar sequências coesas.

# Distillation
A destilação, no contexto de modelos de aprendizado de máquina, refere-se a uma técnica na qual o conhecimento de um modelo maior e mais complexo é transferido para um modelo menor e mais simples. Esse processo é muitas vezes chamado de "destilação do conhecimento" ou "aprendizado por destilação". O principal objetivo é transferir o conhecimento adquirido por um modelo mais complexo para um modelo mais leve, mantendo ou melhorando o desempenho do modelo menor.

### Principais Componentes do Processo de Destilação:

1. **Modelo Professor (Complexo):**
   - Um modelo maior e mais complexo (professor) é treinado em uma tarefa específica. Esse modelo geralmente tem uma capacidade de representação mais rica e é capaz de aprender padrões complexos nos dados.

2. **Modelo Aluno (Simplificado):**
   - Um modelo menor e mais simples (aluno) é criado para realizar a mesma tarefa. Este modelo é mais leve em termos de parâmetros e complexidade.

3. **Transferência de Conhecimento:**
   - O conhecimento do modelo professor é transferido para o modelo aluno. Isso geralmente é feito ajustando o modelo aluno para imitar as previsões do modelo professor.

4. **Regularização:**
   - Técnicas de regularização são frequentemente aplicadas para evitar que o modelo aluno se ajuste excessivamente aos dados de treinamento. Isso pode incluir penalidades em divergências entre as distribuições de probabilidade das previsões do professor e do aluno.

### Benefícios da Destilação:

1. **Redução de Recursos:**
   - Modelos menores resultantes da destilação geralmente têm menos parâmetros, ocupam menos espaço em memória e são mais rápidos para inferência.

2. **Generalização Melhorada:**
   - A destilação pode ajudar o modelo menor a generalizar melhor em relação a dados não vistos, incorporando o conhecimento aprendido pelo modelo professor.

3. **Transferência de Tarefas:**
   - Os modelos destilados podem ser mais eficientes em transferir conhecimento para tarefas relacionadas ou domínios semelhantes.

### Performace
A performance de modelos destilados pode ser avaliada em relação a diferentes métricas e considerações, dependendo do contexto específico da aplicação. Aqui estão algumas considerações gerais sobre a performance de modelos destilados:

1. **Eficiência e Inferência Rápida:**
   - Um dos principais objetivos ao destilar conhecimento é criar modelos mais leves e eficientes, especialmente em termos de inferência. Modelos destilados geralmente apresentam tempos de inferência mais rápidos em comparação com modelos mais complexos, tornando-os adequados para implantação em dispositivos com recursos limitados.

2. **Redução de Parâmetros:**
   - Modelos destilados geralmente têm um número menor de parâmetros em comparação com seus modelos professores mais complexos. Isso pode resultar em menor uso de memória, tornando-os mais escaláveis e eficientes em termos de recursos.

3. **Transferência de Tarefas e Generalização:**
   - A performance de modelos destilados muitas vezes é avaliada em tarefas específicas para as quais foram treinados, mas também pode ser interessante avaliar sua capacidade de generalização. Modelos destilados são projetados para transferir conhecimento para tarefas relacionadas, portanto, avaliar sua performance em uma gama mais ampla de tarefas é relevante.

4. **Conservação de Conhecimento:**
   - A performance é frequentemente avaliada em termos da capacidade do modelo aluno em conservar o conhecimento do modelo professor. Isso pode ser medido usando métricas de desempenho específicas para a tarefa em questão.

5. **Robustez e Regularização:**
   - Modelos destilados são muitas vezes treinados com técnicas de regularização para evitar ajuste excessivo aos dados de treinamento. Avaliar a robustez do modelo destilado em relação a dados não vistos ou condições adversas pode ser crucial.

6. **Comparação com Modelos Base:**
   - A performance de modelos destilados é frequentemente comparada com modelos base que não passaram pelo processo de destilação. Isso ajuda a entender os benefícios e possíveis compensações introduzidos pela destilação.

7. **Avaliação de Tarefas Específicas:**
   - A performance real de modelos destilados é fortemente dependente da tarefa específica para a qual foram treinados. Portanto, métricas relevantes para a tarefa, como acurácia, precisão, recall, F1-score, BLEU score (em tradução automática), entre outras, são utilizadas.

Em resumo, a avaliação da performance de modelos destilados é multifacetada e depende das metas específicas da aplicação. Ela não se limita apenas ao desempenho em tarefas individuais, mas também considera fatores como eficiência computacional, generalização e conservação de conhecimento do modelo professor.

# Current trends
1. As arquiteturas autorregressivas parecem ter assumido o controle, possivelmente apenas porque o campo está focado na geração.

2. Os modelos bidirecionais ainda podem ter vantagem quando se trata de representação. A representação de sentença de modelos bidirecionais como o BERT e como elas se relacionam é superior, eles ainda têm vantagem sobre modelos como GPT.

3. Seq2seq ainda é uma escolha dominante para tarefas com essa estrutura. Eles têm uma vantagem em termos de viés arquitetônico que os ajuda a compreender a tarefa por si próprios.

4. As pessoas ainda estão obcecadas com o crescimento exponencial do número de parâmetros de modelos, mas estamos vendo um movimento contrário em direção a modelos "menores" (parâmetros ainda 10B)

# Information Retrieval
Em Information Retrieval (Recuperação de Informações), "knowledge-intensive tasks" (tarefas intensivas em conhecimento) referem-se a atividades que requerem um entendimento mais profundo e contextual das informações do que simples processamento de texto bruto. Essas tarefas frequentemente envolvem a aplicação de conhecimento prévio, semântica e compreensão mais avançada das relações entre conceitos. 

1. **Question Answering (QA - Resposta a Perguntas):**
   - *Descrição:* Esta tarefa envolve o desenvolvimento de modelos capazes de responder a perguntas formuladas em linguagem natural. O objetivo é compreender a pergunta e fornecer uma resposta precisa.
   - *Exemplo:* Pergunta: "Quem foi o primeiro presidente dos Estados Unidos?" Resposta: "George Washington."

2. **Claim Verification (Verificação de Afirmativas):**
   - *Descrição:* Nesta tarefa, o objetivo é verificar a veracidade de alegações ou afirmações feitas em documentos. Os modelos devem avaliar se uma afirmação é verdadeira, falsa ou não verificável.
   - *Exemplo:* Afirmativa: "O aquecimento global é causado principalmente pela atividade humana." Verificação: verdadeiro.

3. **Commonsense Reasoning (Raciocínio de Senso Comum):**
   - *Descrição:* Envolve a capacidade de raciocinar sobre situações cotidianas usando conhecimento de senso comum. Os modelos precisam inferir conclusões lógicas a partir de informações implícitas.
   - *Exemplo:* Pergunta: "O que acontece quando alguém joga um objeto para cima?" Resposta: "Ele eventualmente cai de volta devido à gravidade."

4. **Long-form Reading Comprehension (Compreensão de Leitura em Longo Formato):**
   - *Descrição:* Similar à leitura de curto formato, mas com documentos mais extensos. Os modelos precisam extrair informações relevantes de textos mais longos.
   - *Exemplo:* Compreender um artigo acadêmico extenso sobre um tópico específico.

5. **Information-Seeking Dialogue (Diálogo em Busca de Informações):**
   - *Descrição:* Esta tarefa envolve a interação em linguagem natural para buscar informações específicas. Os modelos devem entender as consultas do usuário e fornecer respostas relevantes.
   - *Exemplo:* Um usuário pergunta: "Quais são os benefícios para a saúde do consumo de chá verde?" O modelo fornece informações sobre os benefícios do chá verde.

6. **Summarization (Sumarização):**
   - *Descrição:* Envolve condensar informações extensas em um formato mais conciso, mantendo as ideias principais. Pode ser abstrativo (gerando novas frases) ou extrativo (selecionando frases existentes).
   - *Exemplo:* Resumir um artigo de notícias em alguns parágrafos.

7. **Natural Language Inference (Inferência de Linguagem Natural):**
   - *Descrição:* Nesta tarefa, os modelos precisam inferir relações lógicas entre pares de sentenças, determinando se uma sentença é uma implicação, contradição ou neutral em relação à outra.
   - *Exemplo:* Dadas duas sentenças - "O céu é azul" e "Está chovendo" - determinar a relação (por exemplo, contraditório).

Essas tarefas representam desafios significativos em processamento de linguagem natural, e avanços nelas contribuem para a melhoria de sistemas de recuperação de informação.

## Classic Information Retrieval
A Recuperação Clássica de Informações (IR) refere-se aos métodos tradicionais e abordagens utilizados para encontrar informações relevantes a partir de grandes conjuntos de dados, geralmente textuais. Essa disciplina é fundamental em sistemas que buscam organizar, armazenar e recuperar informações de maneira eficiente. Aqui estão alguns dos principais conceitos e componentes da Recuperação Clássica de Informações:

1. **Modelo Booleano:**
   - *Descrição:* Baseado na lógica booleana, este modelo trata a informação como sendo representada por conjuntos de termos e utiliza operadores lógicos (AND, OR, NOT) para expressar relações entre esses termos.
   - *Exemplo:* Busca por "informação AND retrieval" para encontrar documentos que contenham ambas as palavras.

2. **Modelo Vetorial:**
   - *Descrição:* Representa documentos e consultas como vetores em espaços multidimensionais. A similaridade entre vetores é usada para classificar a relevância dos documentos em relação à consulta.
   - *Exemplo:* O algoritmo de similaridade cosseno mede a similaridade entre a consulta e um documento.

3. **TF-IDF (Term Frequency-Inverse Document Frequency):**
   - *Descrição:* Uma técnica que avalia a importância de uma palavra em um documento em relação ao seu uso em toda a coleção de documentos. Termos raros e específicos geralmente recebem maior peso.
   - *Exemplo:* Palavras frequentes em um documento, mas raras na coleção, têm pontuações TF-IDF mais altas.

4. **Modelo Probabilístico:**
   - *Descrição:* Utiliza conceitos probabilísticos para calcular a relevância de um documento em relação a uma consulta. O modelo considera a probabilidade de um documento ser relevante ou não dada a consulta.
   - *Exemplo:* O modelo BM25 é um exemplo de modelo probabilístico comumente utilizado.

5. **Recuperação Booleana:**
   - *Descrição:* Método que lida com consultas expressas usando operadores booleanos, onde os resultados são conjuntos de documentos que satisfaçam as condições da consulta.
   - *Exemplo:* Consulta booleana "information AND retrieval" retorna documentos que contenham ambas as palavras.

6. **Índice Invertido:**
   - *Descrição:* Uma estrutura de dados que armazena informações sobre a ocorrência de termos em documentos. Facilita a rápida identificação de documentos relevantes para uma consulta.
   - *Exemplo:* Um índice invertido pode indicar quais documentos contêm a palavra-chave "informação".

7. **Medidas de Avaliação:**
   - *Descrição:* Métodos para avaliar o desempenho dos sistemas de recuperação, incluindo métricas como precisão, revocação e F1-score.
   - *Exemplo:* A precisão mede a proporção de documentos recuperados que são relevantes.

A Recuperação Clássica de Informações estabeleceu os fundamentos para muitos dos desenvolvimentos modernos em motores de busca e sistemas de processamento de linguagem natural, oferecendo uma base sólida para a organização e busca eficiente de informações.

## Neural Information Retrieval
A Recuperação de Informações Neural refere-se à aplicação de técnicas de redes neurais no contexto da recuperação de informações. Essa abordagem tem ganhado destaque devido à capacidade das redes neurais de aprender representações complexas e contextuais a partir de dados brutos. Aqui estão alguns conceitos e componentes-chave da Recuperação de Informações Neural:

1. **Embedding de Consulta e Documento:**
   - *Descrição:* Representação vetorial densa de consultas e documentos usando técnicas de embedding. Essas representações aprendidas capturam semântica e relações semânticas entre termos.
   - *Exemplo:* Utilização de embeddings de palavras ou subpalavras para representar consultas e documentos.

2. **Modelos de Atenção:**
   - *Descrição:* Modelos que atribuem pesos diferentes a partes específicas da entrada, permitindo que a rede preste mais atenção a certos elementos. São valiosos para lidar com partes importantes de documentos longos.
   - *Exemplo:* O Transformer, um tipo de arquitetura que utiliza mecanismos de atenção, é frequentemente empregado em Recuperação de Informações Neural.

3. **Redes Siamesas:**
   - *Descrição:* Arquiteturas que compartilham parâmetros entre duas redes idênticas. Podem ser usadas para aprender a similaridade entre pares de consultas e documentos.
   - *Exemplo:* Uma rede siamesa pode aprender a medir a similaridade entre a consulta e o documento.

4. **Aprendizado por Transferência:**
   - *Descrição:* Utilização de modelos pré-treinados em grandes conjuntos de dados para tarefas específicas, seguido por ajustes finos em conjuntos de dados menores e mais específicos.
   - *Exemplo:* Um modelo pré-treinado em grandes corpora pode ser ajustado para tarefas específicas de recuperação de informações.

5. **Recuperação de Passagens (Passage Retrieval):**
   - *Descrição:* Em vez de recuperar documentos inteiros, esse enfoque visa identificar passagens relevantes dentro de documentos. Reduz o espaço de busca e foca na informação mais pertinente.
   - *Exemplo:* Recuperação de passagens relevantes em artigos científicos para uma determinada consulta.

6. **Redes Recorrentes e LSTM (Long Short-Term Memory):**
   - *Descrição:* Arquiteturas que mantêm uma memória de longo prazo, permitindo a captura de dependências temporais em dados sequenciais, como texto.
   - *Exemplo:* Uma LSTM pode ser usada para modelar a dependência de palavras em uma sequência de consulta.

7. **Avaliação por Embeddings:**
   - *Descrição:* Medidas de avaliação específicas para avaliar a qualidade dos embeddings gerados pelos modelos, como MAP (Mean Average Precision) e NDCG (Normalized Discounted Cumulative Gain).
   - *Exemplo:* Medir a precisão média da ordem de documentos recuperados pela relevância.
8. **Dense Passage Retriever (DPR):**
   - Descrição: O DPR é uma técnica específica de recuperação de passagens que utiliza uma abordagem densa para representar e recuperar passagens relevantes. Em vez de depender apenas de embeddings de palavras ou subpalavras, o DPR emprega embeddings densos para representar
   passagens inteiras, tornando-o eficiente para a recuperação de informações em grandes conjuntos de dados. É escalável mas limitado nas interações entre consulta (query) e documentos
   - Exemplo: Utilizando embeddings densos para representar passagens em artigos científicos, facilitando a recuperação eficiente de passagens relevantes para uma consulta.

9. **Cross-Encoders:**
   - Descrição: Os Cross-Encoders são arquiteturas que processam simultaneamente a consulta e o documento em uma única etapa, permitindo que a rede capture interações complexas entre ambos. Isso é diferente dos modelos que primeiro geram embeddings separados para a consulta e o documento e, em seguida, calculam a similaridade. Isso pode ser feito a partir do fine tuning do BERT, que acaba causando falta de escalabildiade.
   - Exemplo: Um Cross-Encoder processa simultaneamente uma consulta e um documento, gerando uma pontuação de similaridade diretamente.   

10. ColBERT (Contextualized Late Interaction over BERT):
   - Descrição: ColBERT é um modelo projetado para a recuperação eficiente de documentos em grandes coleções, aproveitando a eficácia do BERT (Bidirectional Encoder Representations from Transformers) para representações contextuais. Ele introduz uma estratégia de interação tardia, onde a interação entre a consulta e o documento é realizada em um estágio posterior, permitindo uma recuperação mais rápida. A inclusão do ColBERT destaca outra inovação notável na Recuperação de Informações Neurais, mostrando como as estratégias de interação e a utilização eficiente de modelos pré-treinados como o BERT podem contribuir para sistemas mais eficazes em ambientes de recuperação de informações.   
   - Representações Contextuais: Utiliza embeddings contextuais do BERT para capturar melhor o significado em contexto de consultas e documentos.
   - Interação Tardia: Diferentemente de outros modelos, adia a interação entre consulta e documento, economizando recursos computacionais durante a fase de recuperação inicial.
   - Estrutura Eficiente: Projetado para escalabilidade, permitindo uma recuperação eficaz em grandes conjuntos de dados.
   - Exemplo: ColBERT pode ser aplicado em motores de busca para recuperar rapidamente documentos relevantes em grandes coleções, utilizando embeddings contextuais para melhorar a compreensão do significado em contexto.

A Recuperação de Informações Neural busca superar as limitações dos métodos clássicos, permitindo uma representação mais rica e adaptativa das informações, especialmente em ambientes de grande escala e complexidade. Essas abordagens têm sido aplicadas com sucesso em sistemas modernos de motores de busca e recomendação.

# Retrieval-augmented in-context learning
"Retrieval-augmented in-context learning" (Aprendizado em Contexto com Recuperação Aprimorada) refere-se a uma abordagem em aprendizado de máquina onde o processo de aprendizado é enriquecido ao incorporar mecanismos de recuperação de informações durante a fase de treinamento do modelo. Essa técnica visa melhorar a capacidade do modelo de compreender e utilizar informações relevantes do contexto, muitas vezes provenientes de grandes conjuntos de dados externos.

1. **Aprendizado em Contexto:**
   - *Descrição:* Refere-se ao treinamento de modelos em ambientes nos quais o contexto é crucial para a compreensão e geração de respostas. Esses modelos geralmente são projetados para tarefas que exigem compreensão de linguagem natural em contextos específicos.

2. **Recuperação Aprimorada:**
   - *Descrição:* Envolve a incorporação de mecanismos de recuperação de informações durante o treinamento do modelo. Isso pode incluir o acesso a grandes bases de dados externas para recuperar informações relevantes durante o processo de aprendizado.

3. **Retrieval-augmented Embeddings:**
   - *Descrição:* Os embeddings (representações vetoriais) usados pelo modelo são aprimorados por meio de informações recuperadas durante o treinamento. Isso permite que o modelo tenha acesso a conhecimentos externos e contextos mais amplos.

4. **Interação com Dados Externos:**
   - *Descrição:* Durante o treinamento, o modelo é capaz de interagir com conjuntos de dados externos, recuperando informações relevantes para aprimorar seu entendimento do contexto em que está operando.

5. **Melhoria da Generalização:**
   - *Descrição:* Espera-se que a incorporação de informações recuperadas durante o treinamento melhore a capacidade do modelo de generalizar para diferentes contextos e condições não vistas durante o treinamento.

6. **Aplicações em Processamento de Linguagem Natural (NLP) e Diálogo:**
   - *Descrição:* Essa abordagem é frequentemente aplicada em tarefas de NLP e diálogo, onde a compreensão do contexto é crucial. Modelos treinados dessa maneira podem ter um desempenho aprimorado em entender e gerar respostas relevantes.

Em resumo, o "retrieval-augmented in-context learning" é uma estratégia que visa fortalecer modelos de aprendizado de máquina, permitindo que eles incorporem informações relevantes de contextos mais amplos durante o processo de treinamento. Isso é particularmente útil em tarefas que exigem uma compreensão profunda do contexto para fornecer respostas precisas e relevantes.

# In Context Learning
## Origins
- **ChomskyBot:** O termo "ChomskyBot" refere-se à influência da teoria linguística de Noam Chomsky nas origens do aprendizado "in context". Chomsky introduziu a ideia de que a compreensão da linguagem envolve uma gramática inata, e isso influenciou pesquisas na criação de modelos que consideram o contexto linguístico ao aprender e compreender informações.

- **n-gram LMs (Modelos de Linguagem n-gram):** Modelos de linguagem n-gram são fundamentais para o entendimento de sequências de palavras. A origem do aprendizado "in context" inclui o desenvolvimento e aprimoramento de modelos n-gram, que consideram a probabilidade de uma palavra dada a sequência anterior de n-1 palavras. Embora simples, esses modelos introduziram a ideia de levar em conta o contexto local para a compreensão da linguagem.

- **decaNLP:**  O decaNLP é um benchmark multitarfa em processamento de linguagem natural que inclui uma variedade de tarefas, como tradução automática, resolução de coreferência, e muitas outras. O desenvolvimento do decaNLP influenciou a pesquisa em modelos que podem realizar várias tarefas em contextos diversos, promovendo a ideia de aprendizado "in context" para lidar com a complexidade do processamento de linguagem natural.

- **Tentativas de Prompt-based Experiments com GPT (Radford et al. 2019):**  O trabalho de Radford et al. em 2019, referenciado aqui, faz parte das origens do aprendizado "in context". As tentativas de experimentos baseados em prompt com o GPT (Generative Pre-trained Transformer) envolvem o uso de modelos de linguagem pré-treinados, como GPT, para responder a perguntas ou completar prompts. Isso destaca a busca por modelos que possam entender e responder a consultas específicas em contextos variados.

O aprendizado "in context" tem raízes em teorias linguísticas, modelos de linguagem tradicionais, benchmarks multitarfa e experimentos com modelos pré-treinados. Essas origens refletem a evolução contínua da pesquisa em processamento de linguagem natural em direção à compreensão mais profunda e contextual da linguagem.

## Core Concepts
### Terminology
- **In-context learning**: É um método de "prompt engineering" em que a task é demonstrada e inputada junto do prompt. Um modelo de linguagem congelado (frozen) que performa uma tarefa apenas condicionando um __prompt__. Sendo congelado, não existem atualização dos gradientes, e o unico aprendizado se dá pelo input textual, que põe o modelo em um estado temporário. Refere-se a abordagens de aprendizado de máquina que consideram e utilizam informações contextuais ao realizar tarefas específicas. Em contextos de processamento de linguagem natural, isso implica compreender o significado das palavras ou frases em relação ao contexto mais amplo em que estão inseridas. Contexto é a informação circundante que afeta a interpretação de uma palavra, frase ou documento. O contexto pode ser local (próximo imediatamente) ou global (abrangendo um conjunto maior de informações).

- **Few-shot in-context learning**: O prompt inclui exemplos do comportamento esperado, e nenhum exemplo do comportamento esperado visto no treinamento.

- **Zero-shot in-context learning**: O prompt não inclui exemplos do comportamento esperado, e nenhum exemplo do comportamento esperado visto no treinamento

- **Autoregressive Training with Teacher Forcing**: O treinamento autoregressivo com forçamento de professor (autoregressive training with teacher forcing) é uma técnica usada no treinamento de modelos de sequência, especialmente em modelos gerativos de linguagem, como os baseados em arquiteturas de Transformers.

1. **Treinamento Autoregressivo:**
   - No contexto de modelos de linguagem, **um modelo é considerado autoregressivo quando a previsão de cada elemento em uma sequência depende dos elementos anteriores da sequência**. Em outras palavras, o modelo gera um elemento por vez, levando em consideração os elementos já gerados.

2. **Forçamento de Professor (Teacher Forcing):**
   - O forçamento de professor é uma técnica durante o treinamento em que, ao ensinar um modelo a gerar sequências, você força o modelo a usar como entrada os elementos verdadeiros (ou seja, corretos) da sequência durante o treinamento, em vez de usar as previsões geradas pelo próprio modelo. Isso é frequentemente utilizado em modelos autoregressivos para acelerar o treinamento e melhorar a estabilidade inicial do modelo.

Portanto, no "treinamento autoregressivo com forçamento de professor", o modelo é treinado para gerar sequências elemento por elemento, onde cada elemento é condicional aos elementos anteriores, e durante o treinamento, a sequência verdadeira (fornecida pelo "professor") é usada como entrada para o modelo. Essa abordagem ajuda a acelerar a convergência do modelo durante o treinamento, permitindo que ele aprenda mais rapidamente a estrutura e a distribuição dos dados sequenciais. No entanto, durante a geração real (quando o modelo é usado para criar sequências), o forçamento de professor não é utilizado, e o modelo gera cada elemento com base nas previsões anteriores.

- **Generation**: Os modelos de linguagem congelados (frozen language models) são modelos pré-treinados que tiveram seus parâmetros fixados e não estão mais sendo atualizados durante o treinamento adicional. Quando se fala em "gerar novos dados" com esses modelos, geralmente está se referindo à capacidade desses modelos de gerar sequências de texto com base no conhecimento que adquiriram durante o treinamento prévio.

A geração de novos dados por modelos de linguagem congelados ocorre por meio de amostragem ou feixe de busca (beam search):

1. **Amostragem (Sampling):**
   - Nesse método, o modelo gera cada palavra ou token da sequência de forma estocástica, ou seja, de maneira probabilística. Cada token é amostrado de acordo com as probabilidades preditas pelo modelo. Isso permite uma certa dose de aleatoriedade na geração, resultando em diferentes possíveis continuations para uma dada entrada.

2. **Feixe de Busca (Beam Search):**
   - No método de feixe de busca, o modelo gera várias sequências simultaneamente e mantém um conjunto de "feixes" (beams) das sequências mais prováveis. Em cada passo, o modelo avalia as probabilidades das palavras seguintes para cada feixe, selecionando as melhores opções. Isso ajuda a produzir sequências mais coerentes e fluentes, uma vez que considera uma busca mais ampla no espaço de possíveis continuations.

É importante notar que, embora os modelos de linguagem congelados possam gerar novas sequências de texto, a qualidade e a relevância dessas sequências dependerão da qualidade do treinamento prévio do modelo. Além disso, a aleatoriedade introduzida pela amostragem ou as escolhas determinísticas do feixe de busca podem resultar em variações nas sequências geradas a cada vez que o modelo é usado para geração.

## Current Movement (2023)
### Dataset used for self-supervision
1. OpenBookCorpus (Bandy and Vincent 2021): https://huggingface.co/datasets/bookcorpusopen
2. The Pile (Gao et al. 2020): https://pile.eleuther.ai
3. Big Science Data (Laurençon et al. 2022): https://huggingface.co/bigscience-data
4. Wikipedia processing: https://github.com/attardi/wikiextractor
5. Pushshift Reddit Data (Baumgartner et al. 2020): https://files.pushshift.io/reddit/
6. Colossal Clean Crawled Corpus (C4; Dodge et al. 2021): https://github.com/allenai/allennlp/discussions/5056

### Dataset used for instruction fine-tuning
- Não sabemos muito sobre o que os laboratórios industriais
estão fazendo aqui.
- Podemos inferir que eles estão pagando muitas pessoas para
gerar dados de instrução.
- Também podemos inferir que eles estão usando seus próprios
modelos para gerar exemplos e julgar
entre exemplos.
- O Stanford Human Preferences Dataset (SHP) (SHP) é um
recurso para ajuste de instrução naturalista (naturalistic fine-tuning): https://huggingface.co/datasets/stanfordnlp/SHP

### Self-instruct
<img src="self_instruct.png">

Self-istruct (learning) é uma técnica para melhorar a performace do modelo utilizando modelos. Nesse caso, criando mais tasks para fazer o instruct fine tuning. Nesse pipeline exemplo, um conjunto de tasks escrita por humanos é armazenada num pool, em que no primeiro passo, o modelo de linguagem cria novas instruções via in-context learning, no passo seguinte a nova instrução é inputada novamente no modelo de linguagem, com um novo prompt, para decidir se a instrução é uma tarefa de classificação ou não. Esse novo par de input/output é usado para o aprendizado supervisionado seguinte. 

<img src="self_instruct_prompt_template.png">

Essa abordagem foi utilizada no modelo Alpaca, possibilitando a especialização do modelo, num tamanho consideravelmente menor e mantendo a performace comparando com os modelos anteriores.

<img src="alpaca.png">

## Techniques and Suggested Methods
### Demonstrations
Demonstration é uma técnica de in-context learning em que se cria recursivamente contexto, pergunta e resposta para se ter respostas corretas.

<img src="demonstration.png">

### Choosing demonstrations
- Podem ser escolhidos aleatoriamente a partir dos dados disponíveis.
- Escolhidos com base no relacionamento com o exemplo alvo.
  - Geração: Recuperada com base na similaridade com a entrada alvo.
  - Classificação: Escolhido para ajudar o modelo a determinar implicitamente o tipo de entrada de destino.
- Filtrando para aqueles que atendem a critérios específicos:
  - Geração: A evidência contém a saída.
  - Geração: O LM prevê a saída correta.
  - Classificação: Todos os rótulos representados.
- Amostrado e reescrito pelo LM:
  - Sintetize múltiplas demonstrações iniciais em demonstrações individuais.
  - Altere o estilo ou a formatação para corresponder ao alvo.

**Seu prompt pode deve conter substrings que foram geradas por um prompt diferente do seu LM.**

### Chain of Thought
Chain of Thought é uma técnica que explica explicitamente o passo a passo do **raciocínio** para responder coisas complexas.

<img src="chain_of_tought.png">

### Generic step-by-step with instruction
É parecido com Chain of Thought, mas aqui, o modelo é instruído a seguir um passo a passo para chegar a uma resposta.

<img src="generic_step.png">

### Self-Consistency
Self-Consistency lembra random forest, no sentido de que são gerados um conjunto de respostas seguindo um raciocínio (ex. chain of tought), e em seguida os caminhos de raciocínio (reasoning paths) são "marginalizados", agregando e selecionando a resposta mais frequente.

<img src="self_consistency.png"> 

### Self-Ask
Através de demonstrações, direcionar o modelo decompor o raciocínio em partes menores em um conjunto de diferentes perguntas para buscar a resposta. Assim, chegando iterativamente na resposta.

<img src="self_ask.png"> 

# Behavorial Evaluation of NLU Models
## Overview
- Este tópico fornece uma visão geral da avaliação comportamental de modelos de compreensão de linguagem natural (NLU). Essa abordagem de avaliação visa examinar o desempenho (comportamento) dos modelos em tarefas específicas relacionadas à compreensão e interpretação de linguagem natural.
- Essa abordagem comportamental foca no output dado certo input, se produziu o resultado esperado ou não.

- Standard evaluations
  - Crie um conjunto de dados a partir de um único processo (mesmo dataset para treino e teste).
  - Divida o conjunto de dados em conjuntos separados de treinamento e teste e reserve o conjunto de teste.
  - Desenvolva um sistema no trem.
  - Somente após a conclusão de todo o desenvolvimento, avalie o sistema com base na precisão do conjunto de testes.
  - Relate os resultados como fornecendo uma estimativa da capacidade de generalização do sistema.

- Adversarial Evaluation
  - Crie um conjunto de dados da maneira que desejar.
  - Desenvolva e avalie o sistema usando esse conjunto de dados, de acordo com os protocolos que você escolher.
  - Desenvolva um novo conjunto de dados de teste com exemplos que você suspeita ou sabe que serão desafiadores, considerando seu sistema e o conjunto de dados original.
  - Somente após a conclusão de todo o desenvolvimento do sistema, avalie o sistema com base na precisão do novo conjunto de dados de teste.
  - Relate os resultados como fornecendo uma estimativa da capacidade de generalização do sistema.

- Winograd sentences
As "Winograd Schemas" são uma forma de teste projetada para avaliar a compreensão semântica e o raciocínio lógico dos sistemas de processamento de linguagem natural (PLN), especialmente em relação à resolução de ambiguidades pronominais. Esses testes foram propostos por Terry Winograd em 1972 como uma maneira de avaliar a compreensão de máquinas em relação a questões de ambiguidade e raciocínio sobre o significado de pronomes em contextos específicos.

Cada "schema" consiste em uma sentença curta que envolve uma ambiguidade pronominal. Aqui está um exemplo clássico de Winograd Schema:

  - The trophy doesn’t fit into the brown suitcase because
it’s too small. What is too small? The suitcase / The trophy
  - The trophy doesn’t fit into the brown suitcase because it’s too large. What is too large? The suitcase / The trophy
  - The council refused the demonstrators a permit because they feared violence. Who feared violence? The council / The demonstrators
  - The council refused the demonstrators a permit because they advocated violence. Who advocated violence? The council / The demonstrators
  - O troféu não cabe na mala marrom porque
é muito pequeno. O que é muito pequeno? A mala / O troféu
  - O troféu não cabe na mala marrom porque é muito grande. O que é muito grande? A mala / O troféu
  - O conselho recusou autorização aos manifestantes porque temiam violência. Quem temia a violência? O conselho / Os manifestantes
  - O conselho recusou autorização aos manifestantes porque eles defendiam a violência. Quem defendeu a violência? O conselho / Os manifestantes

Esses testes são projetados para serem desafiadores para os sistemas de PLN, pois exigem um entendimento mais profundo do significado das palavras e da estrutura da sentença. Avaliar corretamente esses casos requer algum nível de raciocínio semântico e conhecimento contextual.

Os "Winograd Schemas" têm sido usados como uma métrica de avaliação em pesquisas que visam medir a capacidade de máquinas em compreender contextos, fazer inferências e resolver ambiguidades. Eles são particularmente relevantes para destacar as limitações dos sistemas de PLN em situações em que o conhecimento contextual e o raciocínio lógico são essenciais para uma compreensão correta.

- Levesque’s (2013) adversarial framing
  - Poderia um crocodilo correr uma corrida de obstáculos? “A intenção aqui é clara. A questão pode ser respondida pensando bem: um crocodilo tem pernas curtas; as sebes numa corrida de obstáculos seriam demasiado altas para o crocodilo saltar; então não, um crocodilo não pode correr com obstáculos.”
  - Frustrar truques baratos: “Podemos encontrar questões em que truques baratos como este não serão suficientes para produzir o comportamento desejado? Infelizmente, isso não tem uma resposta fácil. O melhor que podemos fazer, talvez, é elaborar cuidadosamente um conjunto de questões de múltipla escolha e depois estudar os tipos de programas de computador que possam ser capazes de respondê-las.”

## Analytical Considerations
- Aborda considerações analíticas relevantes para avaliações comportamentais de modelos NLU. Isso pode incluir a escolha de métricas, a seleção de conjuntos de dados apropriados e a formulação de estratégias para compreender a eficácia dos modelos em contextos específicos.

- Limits of behavorial testing:
Os limites dos testes adversariais em avaliações comportamentais de modelos de linguagem incluem:
  - Dificuldade em gerar amostras realistas: é desafiador criar um conjunto de dados que seja semanticamente válido e sintaticamente correto, mas também suficientemente enganoso para induzir o modelo a cometer erros. Isso pode resultar em amostras artificiais ou irrelevantes que não refletem situações do mundo real.
  - Falta de contexto: muitas vezes, as interações humanas envolvem contexto complexo e dinâmico que é difícil de ser capturado por meio de testes adversariais automatizados. Isto pode levar a uma subestimação da capacidade do modelo em entender e processar informação em diferentes cenários.
  - Limitações na diversidade das tarefas: atualmente, existem poucas tarefas bem-definidas e amplamente aceitas para avaliar modelos de linguagem. Como resultado, os testes adversariais geralmente se concentram em um pequeno número dessas tarefas, o que pode limitar sua abrangência e generalização.
  - Escalabilidade: à medida que os modelos de linguagem aumentam de tamanho e sofisticação, torna-se cada vez mais computacionalmente inviável executar testes adversariais detalhados sobre todos os aspectos do modelo. Além disso, é possível que alguns métodos de teste simplesmente não escalonem para modelos muito grandes.
  - Interpretação dos resultados: é frequentemente difícil interpretar os resultados dos testes adversariais devido à falta de comparação com outros modelos ou métricas estabelecidas. Isto pode levar a conclusões equivocadas sobre as verdadeiras habilidades e deficiências dos modelos de linguagem.
  - Desequilíbrio entre risco e recompensa: ao longo do desenvolvimento de modelos de linguagem, existe um potencial de uso indevido ou malicioso deles. Testes adversariais podem exacerbar esses riscos ao revelar vulnerabilidades exploráveis no sistema, mas pouca pesquisa tem sido dedicada a mitigar esse problema.
  - Ausência de padronização: há pouca padronização nas metodologias usadas para testes adversariais em avaliações comportamentais de modelos de linguagem, o que dificulta a comparação direta dos resultados obtidos por diferentes equipes de pesquisadores.

- Metrics
 Existem vários limites associados à utilização de métricas baseadas em acurácia em avaliações comportamentais de modelos de linguagem: 
  - Ignora significado e contexto: métricas baseadas em precisão e recall normalmente consideram apenas correspondências exatas entre previsões e respostas corretas, ignorando assim nuances de significado e contexto. Essa abordagem pode levar a uma visão estreita e superficial das habilidades de compreensão e geração de linguagem pelos modelos.
  - Sensibilidade insuficiente a erros específicos: métricas baseadas em acurácia podem não distinguir entre diferentes tipos de erro, mesmo quando eles indicam problemas distintos nos modelos. Por exemplo, confundir "gato" com "cachorro" pode ter consequências diferentes de confundir "gato" com "torniquete".
  - Baixa granularidade: métricas simples como precisão e recall fornecem informações agregadas sobre o desempenho do modelo, mas podem ocultar pormenores importantes sobre seu comportamento em diferentes tarefas ou contextos. Isso pode impedir uma análise cuidadosa e diagnóstico adequado dos pontos fracos do modelo.
  - Dependência excessiva de benchmarks: métricas baseadas em acurácia tendem a concentrar-se em benchmarks pré-existentes, o que pode limitar sua utilidade para novas tarefas ou domínios. Adicionalmente, isso pode incentivar a otimização local em detrimento do desenvolvimento de habilidades de linguagem mais generativas e robustas.
  - Inibição da inovação: foco excessivo em métricas quantitativas pode desencorajar investigação em áreas menos mensuráveis, mas igualmente importantes, como ética, responsabilidade social e inclusividade nos sistemas de linguagem artificial.
  - Avaliação unidimensional: métricas baseadas em acurácia geralmente oferecem uma única dimensão para avaliar o desempenho do modelo, o que pode ser insuficiente para capturar a riqueza e complexidade da linguagem humana. Isso pode resultar em uma subestimação ou overestimation da capacidade do modelo em diferentes aspectos da comunicação natural.
  - Impacto negativo no design de experimentos: métricas simplistas podem conduzir a experiimentos mal projetados, onde as questões são formuladas de maneira a maximizar a pontuação do modelo em vez de avaliar seus verdadeiros limites e capacidades. Em última instância, isso pode levar a conclusões equívocas sobre as habilidades e deficiências dos modelos de linguagem.

- Inoculation by fine-tuning
A técnica de "Inoculation by Fine-Tuning" refere-se a uma abordagem em aprendizado de máquina, especialmente em processamento de linguagem natural (PLN), para melhorar a robustez dos modelos diante de fraquezas específicas identificadas em conjuntos de dados ou em modelos pré-treinados. Essa abordagem envolve a introdução controlada de exemplos "inoculativos" durante a fase de ajuste fino (fine-tuning) para fortalecer o modelo contra deficiências conhecidas.

<img src="inoculation.png">

Aqui estão alguns aspectos chave dessa técnica:

1. **Identificação de Fraquezas:**
   - Antes do processo de "Inoculation by Fine-Tuning", é necessário identificar fraquezas específicas nos dados de treinamento ou no modelo. Isso pode incluir problemas como viés, falta de diversidade, ou sensibilidade a determinados tipos de exemplos.

2. **Inoculação Controlada:**
   - Durante a fase de ajuste fino do modelo, exemplos específicos são adicionados deliberadamente ao conjunto de treinamento para "inocular" ou fortalecer o modelo contra as fraquezas identificadas. Esses exemplos são escolhidos para desafiar o modelo de maneira que aborde as limitações conhecidas.

3. **Ampliação da Diversidade:**
   - A inoculação pode envolver a inclusão de exemplos que abordem a diversidade, tornando o modelo mais robusto a diferentes contextos e perspectivas. Isso ajuda a mitigar o risco de overfitting a características específicas do conjunto de treinamento original.

4. **Aprimoramento da Generalização:**
   - A ideia fundamental é que a introdução controlada de exemplos desafiadores durante o ajuste fino pode levar a um modelo mais robusto, capaz de generalizar melhor para situações diversas, mesmo aquelas que podem não ter sido bem representadas no conjunto de dados original.

5. **Detecção de Fraquezas:**
   - Além de fortalecer o modelo, essa abordagem também pode ser usada para identificar fraquezas residuais. Monitorando o desempenho do modelo em exemplos inoculativos, os desenvolvedores podem avaliar se as fraquezas estão sendo efetivamente abordadas ou se novas fraquezas são descobertas.

Essa técnica é particularmente relevante em situações em que a robustez do modelo é uma preocupação e onde se deseja mitigar potenciais vieses, falhas ou limitações do conjunto de dados ou do modelo. A "Inoculation by Fine-Tuning" destaca a importância de um ajuste fino estratégico para fortalecer modelos e melhorar seu desempenho em cenários mais desafiadores.

## Compositionality
- Explora o conceito de composicionalidade na avaliação de modelos NLU. A composicionalidade refere-se à capacidade de um modelo entender e compor significados complexos a partir de partes menores. Esse tópico analisa como os modelos se saem em tarefas que exigem compreensão e manipulação de significados compostos.

**composicionalidade**

- **Definição:** A composicionalidade é um princípio fundamental em linguística e processamento de linguagem natural. Refere-se à ideia de que o significado de uma expressão complexa é determinado pela combinação e interação dos significados de suas partes constituintes. Em outras palavras, o significado de uma frase ou expressão é construído a partir dos significados de suas palavras e das relações sintáticas entre elas.

- **Exemplo:** Considere a frase "O gato está no telhado". A composicionalidade sugere que o significado da frase é construído a partir do significado de cada palavra individual ("gato", "telhado", "no", "está") e das relações sintáticas entre elas.

- **Importância:** A composicionalidade é crucial para a compreensão de linguagem natural e é um princípio subjacente em muitos modelos de processamento de linguagem natural. Modelos que capturam efetivamente a composicionalidade são capazes de generalizar para novas expressões e estruturas, pois entendem como as partes se combinam para formar significados mais complexos.

**Systematicity:**

- **Definição:** A systematicity refere-se à capacidade de um sistema cognitivo, como o cérebro humano ou modelos de linguagem, de exibir padrões sistemáticos e consistentes em sua representação e processamento de informações. Em termos de linguagem, isso implica que se um sistema compreende ou gera uma expressão em um contexto, ele deve ser capaz de fazer o mesmo em contextos semanticamente semelhantes.

- **Exemplo:** Se um modelo de linguagem compreende a relação entre "cão" e "latindo" em um contexto, espera-se que ele também compreenda a relação entre "gato" e "miando" em um contexto semelhante.

- **Importância:** A systematicity é uma propriedade desejável em modelos de linguagem, pois reflete a capacidade de generalizar padrões aprendidos para novas situações semelhantes. Modelos que exibem systematicity são mais robustos e capazes de lidar com variações semânticas e estruturais na linguagem natural.

Tanto a composicionalidade quanto a systematicity são conceitos essenciais para o desenvolvimento de modelos de linguagem que possam compreender e gerar textos de maneira mais flexível e generalizada, aproximando-se da capacidade humana de lidar com a complexidade da linguagem natural.

## COGS and ReCOGS
- Introduz os conceitos de COGS (Compositional Generalization Score) e ReCOGS (Reverse Compositional Generalization Score). Essas métricas são utilizadas para avaliar a capacidade de modelos NLU em generalizar para composições inversas ou novas composições, medindo a robustez da compreensão composicional.

<img src="cogs_recogs.png">

**COGS (Compositional Generalization Score):**

- **Definição:** O COGS, ou Compositional Generalization Score, é uma métrica usada para avaliar a capacidade de generalização composicional de modelos de linguagem. Ela mede o quão bem um modelo consegue generalizar para novas composições sintáticas ou semânticas que não foram explicitamente vistas durante o treinamento.

- **Metodologia:** Para calcular o COGS, são criadas novas combinações de palavras ou estruturas sintáticas que não fazem parte do conjunto de treinamento. O modelo é então testado nessas novas composições, e o COGS é calculado com base na capacidade do modelo de compreender e gerar corretamente essas novas combinações.

**ReCOGS (Reverse Compositional Generalization Score):**

- **Definição:** O ReCOGS, ou Reverse Compositional Generalization Score, é uma métrica relacionada ao COGS, mas com uma abordagem ligeiramente diferente. Ele avalia a capacidade de generalização em direção oposta, medindo a capacidade de um modelo de entender composições que envolvem inversões sintáticas ou semânticas em comparação com o treinamento original.

- **Metodologia:** Assim como no COGS, novas combinações são criadas, mas o foco no ReCOGS é avaliar se o modelo é capaz de generalizar bem para composições inversas ou "reversas" que não foram vistas durante o treinamento. Isso inclui situações em que a ordem de palavras ou a estrutura sintática é invertida em comparação com as instâncias de treinamento.

Ambas as métricas, COGS e ReCOGS, são projetadas para avaliar a capacidade de modelos de linguagem de generalizar de maneira composicional, proporcionando uma visão mais aprofundada sobre como esses modelos podem lidar com novas combinações de palavras ou estruturas sintáticas não encontradas durante o treinamento. Essas métricas são especialmente relevantes em tarefas que envolvem compreensão de linguagem natural e geração de texto, onde a capacidade de generalizar é crucial para a robustez do modelo.

## Adversarial Testing
- Explora a prática de realizar testes adversariais para avaliar modelos NLU. Testes adversariais envolvem a criação de exemplos desafiadores que podem expor as vulnerabilidades ou limitações dos modelos, oferecendo uma visão crítica de seu desempenho em situações difíceis.

- **Definição:** O teste adversarial é uma técnica usada para avaliar a robustez de modelos, incluindo modelos de linguagem natural. Envolve a criação de exemplos desafiadores, também conhecidos como exemplos adversariais, projetados para explorar vulnerabilidades ou fraquezas nos modelos.

- **Metodologia:**
  1. **Geração de Exemplos Desafiadores:** Os exemplos adversariais são criados introduzindo pequenas perturbações nos dados de entrada, como alterar palavras, adicionar ruído ou realizar outras modificações sutis.
  
  2. **Avaliação do Comportamento do Modelo:** Os exemplos adversariais são então fornecidos ao modelo para avaliar como ele lida com essas perturbações. A ideia é testar se o modelo mantém o desempenho esperado ou se sua saída é significativamente afetada pelos exemplos desafiadores.

- **Objetivos:**
  - **Expor Fraquezas:** O teste adversarial visa expor fraquezas nos modelos que podem não ser evidentes em avaliações padrão. Isso inclui situações em que o modelo pode falhar ao lidar com entradas inesperadas ou manipuladas.

  - **Avaliar Robustez:** A capacidade de um modelo de lidar com exemplos adversariais é uma medida de sua robustez. Modelos mais robustos são menos propensos a serem enganados ou fornecerem respostas incorretas quando confrontados com entradas desafiadoras.

- **Aplicações:**
  - **Segurança:** O teste adversarial é crucial em domínios onde a segurança é fundamental, como em sistemas de reconhecimento de voz, classificação de imagens, tradução automática e modelos de linguagem natural.

  - **Melhoria Contínua:** Ao identificar as fraquezas dos modelos por meio de testes adversariais, os desenvolvedores podem aprimorar continuamente seus modelos, implementando contramedidas específicas para lidar com exemplos desafiadores.

- **Desafios e Variações:**
  - **Transferência de Ataque:** Em alguns casos, ataques adversariais projetados para um modelo podem ser transferidos para outros modelos, mesmo que não tenham sido treinados com os mesmos dados.

  - **Defesa Adversarial:** Pesquisas também se concentram no desenvolvimento de técnicas de defesa adversarial para tornar os modelos mais resilientes contra exemplos adversariais.

O teste adversarial é uma ferramenta valiosa na avaliação da robustez de modelos de linguagem e em outros domínios da inteligência artificial. Ele desempenha um papel crítico na identificação e mitigação de vulnerabilidades, contribuindo para o desenvolvimento de sistemas mais confiáveis e seguros.  
 
## Adversarial NLI
- Este subtopico específico concentra-se em adversarial Natural Language Inference (NLI). Examina como os modelos se comportam em cenários desafiadores relacionados à inferência de relações lógicas entre sentenças, destacando as nuances e desafios associados a essa tarefa específica.

- **Definição:** Adversarial NLI refere-se a uma abordagem específica de teste adversarial aplicada à tarefa de Inferência de Linguagem Natural (NLI). A NLI envolve determinar a relação lógica entre duas sentenças, geralmente rotuladas como "hipótese" e "premissa", classificando se a hipótese é verdadeira (entailment), falsa (contradiction) ou neutra (neutral) em relação à premissa.

- **Metodologia:**
  1. **Geração de Exemplos Adversariais NLI:** Exemplos adversariais para a tarefa NLI são criados manipulando sutilmente as premissas e hipóteses, introduzindo mudanças que desafiam o modelo, mas que podem parecer plausíveis para um observador humano.

  2. **Avaliação de Desempenho:** Esses exemplos adversariais são então usados para avaliar o desempenho do modelo NLI. O objetivo é testar se o modelo é capaz de manter uma inferência correta mesmo em situações em que as sentenças são cuidadosamente modificadas para induzir erros.

- **Objetivos:**
  - **Identificar Fraquezas:** Adversarial NLI é projetado para identificar fraquezas em modelos NLI, revelando situações em que os modelos podem falhar em realizar inferências lógicas precisas.

  - **Melhorar a Robustez:** Ao expor os modelos a exemplos adversariais, os desenvolvedores podem aprimorar a robustez dos modelos, ajustando-os para lidar melhor com variações sutis nas formulações das sentenças.

- **Desafios Específicos:**
  - **Preservação do Significado:** Ao criar exemplos adversariais, é importante manter o significado original das sentenças. Modificações excessivas podem prejudicar a interpretabilidade do teste.

  - **Transferência de Ataque:** O teste adversarial NLI também pode abordar a questão da transferência de ataque, onde um modelo treinado em um conjunto de dados específico é vulnerável a ataques adversariais transferidos de outros modelos ou domínios.

- **Aplicações:**
  - **Benchmarks de Avaliação:** Adversarial NLI é utilizado como um método adicional de avaliação de modelos NLI, complementando métricas tradicionais e ajudando a fornecer uma visão mais completa do desempenho do modelo.

  - **Desenvolvimento de Modelos Robustos:** Os insights obtidos por meio de testes adversariais podem orientar o desenvolvimento de modelos de linguagem mais robustos, capazes de lidar com variações na formulação de sentenças.

Adversarial NLI é uma estratégia valiosa para avaliar e aprimorar modelos de inferência de linguagem natural, contribuindo para o desenvolvimento de sistemas mais confiáveis e consistentes em sua capacidade de compreender relações lógicas entre sentenças.

## DynaSent
- DynaSent é um dataset utilizado para avaliar modelos em tarefas dinâmicas de análise de sentimentos. Este conjunto de dados dinâmico destaca a importância de compreender como os modelos NLU lidam com a mudança de sentimentos ao longo do tempo e contextos diversos. O DynaSent é construído para testar a capacidade dos modelos de compreender e generalizar em cenários onde os sentimentos podem variar, fornecendo uma visão mais realista das capacidades de compreensão de sentimentos dos modelos em situações dinâmicas e em evolução. A dinamicidade do conjunto de dados apresenta desafios únicos, exigindo que os modelos não apenas identifiquem sentimentos, mas também se adaptem a mudanças de tom e contextos específicos ao longo do tempo. Ao focar no DynaSent, os pesquisadores podem entender melhor como os modelos NLU lidam com nuances temporais e variações de sentimentos, contribuindo para uma avaliação mais completa da capacidade desses modelos em contextos de análise de sentimentos.

# Structural Evaluation of NLU Models
### Overview
Descreve uma visão geral dos métodos de análise utilizados na compreensão de linguagem natural (NLU). Estes métodos visam entender e avaliar o funcionamento interno dos modelos de NLU.

Testes comportamentais tem a limitação de que apenas avaliam o output, não as causas que geraram os outputs, a configuração do modelo não é avaliada. Ou seja, com testes comportamentais não temos uma garantia sistemática de que pra cada string inteira o modelo se comportará como planejado. Para suprir esse gap, testes Estruturais como Probing, Feature Attribution, IIT e DAS foram desenvolvidos e são foco de pesquisa. Os testes estruturais são uma forma de olhar dentro da "black box" dos modelos de linguagem.

Os principais métodos de avaliação estrutural de modelos de NLU que serão estudados são Probing, Feature Attibution, IIT e DAS.

Os métodos podem ser agrupados no framework analítico a seguir. Nele, são representados a capacidade do método de caracterizar as representações (inputs, internas e outputs), garantir afirmações cauais/inferencias causais sobre o modelo, e capacidade de melhoria dos modelos ao utilizar esses métodos

<img src="analytical_framework.png">

### Probing
O Probing é uma técnica utilizada na avaliação de modelos de Compreensão de Linguagem Natural (NLU) para investigar e entender as capacidades linguísticas subjacentes do modelo. Em termos simples, o Probing envolve a introdução (seleção ou criação) de tarefas de avaliação específicas que sondam aspectos linguísticos particulares para determinar o que o modelo aprendeu durante o treinamento. No contexto do Probing, os pesquisadores selecionam ou criam tarefas específicas para avaliar aspectos linguísticos particulares do modelo de Compreensão de Linguagem Natural (NLU). Esses aspectos linguísticos são avaliados utilizandos um modelo menor (probe) e específico para cada tarefa, de forma supervisionada, para determinar o que está latentemente codificado em suas representações ocultas.

Overview:
1. Core idea: use supervised models (the probes) to determine what is latently encoded in the hidden representations of our target models.
2. Often applied in the context of BERTology – see especially Tenney et al. 2019.
3. A source of valuable insights, but we need to proceed with caution: É A very powerful probe might lead you to see things that aren’t in the target model (but rather in your probe).
4. Probes cannot tell us about whether the information that we identify has any causal relationship with the target model’s behavior.

Probing é uma fonte de insights valiosos, mas precisamos proceder com cautela:
- Uma sondagem muito poderosa pode levar você a ver coisas que não estão no modelo de destino (mas sim na sua sondagem).
- As sondagens não podem nos dizer se as informações que identificamos têm alguma relação causal com o comportamento do modelo alvo.

Receita para probing:
1. Estabeleça uma hipótese sobre um aspecto da estrutura interna do modelo alvo.
2. Escolha uma tarefa supervisionada que seja uma proxy da estrutura interna de interesse.
3. Identifique o local do modelo onde você acredita que a estrutura será codificada. Um conjunto de representações vetoriais interna do modelo.
4. Treine a sonda supervisionada no(s) local(is) escolhido(s)

Conneau et al. 2018; Tenney et al. 2019

#### Core Method
<img src="probing_core_method.png">

O processo é feito de forma instrutiva com um modelo como o BERT, ele é rodado milhares de vezes, a representação vetorial escolhida é coletada para cada rodada e usada pra construir um pequeno connjunto de dados de aprendizagem supervisionada. Então um pequeno modelo linear é fitado na representação interna (representação vetorial), usando os rótulos da task escolhida. O modelo BERT foi utilizado somente como um "motor" (engine) para gerar as representações vetoriais de cada rodada e criar o dataset com as representações e a task como label.

O exemplo acima é uma simplificação para fins didáticos.

#### **Processo do Probing:** (Gerado com chatGPT)
1. **Seleção/criação de Tarefas Específicas:** Os pesquisadores escolhem ou projetam tarefas específicas que abordam características linguísticas particulares que desejam avaliar no modelo de NLU. Essas tarefas podem incluir aspectos sintáticos, semânticos, de entidades nomeadas, ou qualquer outra propriedade linguística de interesse. Cada tarefa tem um objetivo claro e específico que ajuda a sondar o conhecimento ou a capacidade do modelo em relação a essa propriedade linguística. Por exemplo, se a tarefa é sobre entidades nomeadas, o objetivo pode ser identificar se o modelo consegue reconhecer e rotular corretamente entidades em uma frase.

2. **Treinamento do Prober (Sonda):** Introdução de um "prober" ou sonda (modelo treinado separadamente para realizar a tarefa específica), que é um modelo simples e específico para a tarefa de avaliação escolhida. Este modelo é treinado para avaliar a habilidade do modelo principal (o modelo de NLU) na tarefa específica.

3. **Avaliação do Modelo de NLU:** O modelo de NLU é avaliado na tarefa de sondagem usando o prober treinado. Isso ajuda a determinar o quanto o modelo principal possui conhecimento ou habilidade na área específica sondada. Isso fornece insights sobre como o modelo principal lida com a tarefa específica, indicando seu nível de conhecimento ou capacidade na propriedade linguística em questão.

#### **Exemplos de Tarefas Probing:**
1. **Sintaxe:**
   - Tarefa: Prever a estrutura sintática de uma sentença.
   - Exemplo: Dada a sentença "O gato está na caixa", prever a árvore sintática.

2. **Semântica:**
   - Tarefa: Avaliar a compreensão semântica.
   - Exemplo: Dada a pergunta "Qual é a capital da França?", prever a resposta "Paris".

3. **Entidades Nomeadas:**
   - Tarefa: Identificar entidades nomeadas.
   - Exemplo: Dada a frase "Barack Obama nasceu em Honolulu", prever "Barack Obama" como uma entidade nomeada.

4. **Concordância de Gênero:**
   - Tarefa: Avaliar a compreensão de gênero.
   - Exemplo: Dada a frase "O médico falou com a paciente. Ele deu conselhos", prever que "Ele" se refere ao médico.

O Probing é uma ferramenta valiosa na avaliação de modelos de NLU, proporcionando uma visão mais detalhada de suas habilidades linguísticas e contribuindo para a compreensão de como esses modelos processam e representam informações linguísticas.

### Feature Attribution
O método de atribuição de características (feature attribution) visa identificar quais partes do texto de entrada contribuem mais para as decisões do modelo. Métodos como Saliency Maps ou LRP (Layer-wise Relevance Propagation) podem ser utilizados para essa análise.

**Métodos de Feature Attribution**:

#### Integrated Gradients
**Integrated Gradients** é uma técnica de atribuição de importância que visa explicar as predições de modelos de aprendizado de máquina, mostrando como cada recurso de entrada contribui para a saída do modelo. Esta técnica tem suas raízes na teoria de integração de cálculo, permitindo uma abordagem sistemática para a atribuição de importância ao longo de uma trajetória contínua entre uma linha de base (geralmente uma entrada nula) e a entrada original.

**Princípios e Axiomas dos Integrated Gradients:**

***Sensitivity***
Se duas entradas x e x 0 diferem apenas na dimensão i e levam a previsões diferentes, então o recurso fi tem atribuição diferente de zero.
```
M([1, 0, 1]) = positivo
M([1, 1, 1]) = negativo
```
***Implementation invariance***
Se dois modelos M e M0 têm comportamento de entrada/saída idêntico, então as atribuições para M e M0 são idênticas.

**Processo de Cálculo:**
1. **Definição da Linha de Base:** Começa-se com uma linha de base, que é uma entrada nula ou uma entrada que representa uma condição de referência. Geralmente, todos os recursos da linha de base são definidos como zero.

2. **Criação de Trajetória:** Cria-se uma trajetória suave entre a linha de base e a entrada original. Isso pode ser realizado por meio de uma interpolação linear ou outra técnica que permita seguir uma trajetória contínua.

3. **Cálculo dos Gradientes:** Calculam-se os gradientes da saída do modelo em relação à entrada em vários pontos ao longo da trajetória. Esses gradientes indicam como cada recurso contribui para a variação na saída.

4. **Integração ao Longo da Trajetória:** Integram-se os gradientes ao longo da trajetória utilizando uma técnica de integração, como a regra do trapézio. Isso resulta nas atribuições de importância integradas para cada recurso.

5. **Atribuição de Importância Final:** A atribuição de importância final para cada recurso é obtida subtraindo a importância na linha de base da importância na entrada real.

**Aplicações:**
Os Integrated Gradients são frequentemente utilizados para interpretar modelos de aprendizado profundo em tarefas de NLU, fornecendo uma compreensão mais refinada de como as características de entrada influenciam as decisões do modelo. Eles são aplicados em várias tarefas, incluindo classificação de texto, processamento de linguagem natural e visão computacional.

#### Outros métodos
Saliency Maps:
- Descrição: Saliency maps destacam as regiões mais importantes nas entradas. Em NLU, isso pode ser aplicado a palavras ou tokens específicos para entender quais contribuem mais para a decisão do modelo.
- Exemplo: Para uma classificação de sentimento, uma saliency map poderia mostrar as palavras mais relevantes que influenciam a predição positiva ou negativa.

Gradient-based Methods:
- Descrição: Métodos baseados em gradientes calculam a derivada da saída do modelo em relação às entradas. Isso indica como pequenas mudanças nas entradas afetam a saída.
- Exemplo: Se uma palavra específica em uma frase tem uma grande influência nas previsões, seu gradiente seria significativo.

LIME (Local Interpretable Model-agnostic Explanations):
- Descrição: LIME cria interpretações locais para as previsões do modelo, gerando instâncias próximas da entrada original e treinando um modelo interpretável nessas instâncias.
- Exemplo: Em NLU, LIME pode gerar frases similares à entrada original e destacar as palavras mais importantes para uma decisão específica do modelo.

Exemplo de um modelo de NLU treinado para classificação de sentimento em análises de produto:
- Entrada: "O produto é incrível, superando minhas expectativas!"
- Saída Prevista: Sentimento Positivo

Usando Feature Attribution:
- Saliency Map: Pode destacar as palavras "incrível" e "superando" como influentes para a predição positiva.
- Gradient-based Method: Mostrará como mudanças nessas palavras afetam a predição positiva.
- LIME: Pode gerar uma frase similar sem a palavra "incrível" e ver como isso afeta a previsão.
- Integrated Gradients: Destacará a importância das palavras ao longo da trajetória da entrada original para uma linha de base.

Essas técnicas fornecem interpretações valiosas sobre como o modelo atribui importância às diferentes partes da entrada, contribuindo para uma compreensão mais profunda de seu comportamento em tarefas específicas de NLU.

### Causal Abstraction & Interchange Intervention Training (IIT)
#### Causal Abstraction
Receita para abstração causal
1. Estabeleça uma hipótese sobre (um aspecto) da estrutura causal do modelo alvo.
2. Procure um alinhamento entre o modelo causal e o modelo alvo.
3. Realize **intervenções de intercâmbio (interchange interventions)**

<img src="causal_abstraction.png">

Neste método, são feitas intervenções alterando os valores dos neurônios para verificar a relação causal a partir do resultado. A partir das intervenções, podemos validar a hipótese gerada da estrutura causal do modelo. Quando a intervenção não causa nenhum impacto, provamos que aquela região não exerce papel causal no comportamento de entrada e saída do modelo.

Como não podemos verificar todas as possibilidades possíveis de intervenção num cenário real, devido a infinidade de combinações possíveis, é feito em um subconjunto de exemplos. E para metrificar o sucesso das intervenções, é aplicado ***interchgange intervention accuracy (IIA)**.

#### interchgange intervention accuracy (IIA)

1. IIA é a percentagem de intervenções de intercâmbio que conduzem a resultados que correspondem aos do modelo causal no alinhamento escolhido.
2. O IIA é dimensionado em [0, 1], como acontece com uma métrica de precisão normal.
3. O AII pode, na verdade, estar acima do desempenho da tarefa, se as intervenções de intercâmbio colocarem o modelo num estado melhor.
4. O IIA é extremamente sensível ao conjunto de intervenções de intercâmbio realizadas.
5. Preste especial atenção a quantas intervenções de intercâmbio devem alterar o rótulo de resultados, uma vez que fornecem as evidências mais claras.

#### Descobertas da abstração causal

1. Modelos de BERT ajustados têm sucesso em exemplos difíceis e fora de domínio, envolvendo implicação e negação lexical, **porque** são abstraídos por programas simples de monotonicidade (Geiger et al. 2020).
2. Modelos BERT ajustados têm sucesso na tarefa MQNLI **porque** encontram soluções composicionais (Geiger et al. 2021).
3. Os modelos têm sucesso na tarefa MNIST Pointer Value Retrieval (MNIST-PVR; Zhang et al. 2021) **porque** são abstraídos por programas simples como “se o dígito for 6, então o rótulo está no canto inferior esquerdo” (Geiger et al. .2021).
4. BART e T5 utilizam representações coerentes de entidades e situações que evoluem à medida que o discurso se desenrola (Li et al. 2021).
5. Este notebook do curso é uma introdução prática a essas técnicas: ```https://github.com/cgpotts/cs224u/blob/main/iit_equality.ipynb```

### Interchange Intervention Training (IIT)
IIT se baseia na abstração causal com interchange intervention, mas os parametros são atualizados com o sinal do erro do modelo em relação ao resultado da hipótese causal. Os pesos são atualizados para ajustar o output em relação ao resultado da hipótese causal. No subset em que houve a intervenção, é feito uma atualização "dupla", porquê é atualizado em relação ao target exemplo e da fonte (a direita). Primeiro recebe os parâmetros da fonte, depois eles são atualizados de acordo com o erro em relação a estrutura causal da hipótese. No fim, o     modelo é forçado a ter a estrutura causal da hipótese criada.

<img src="iit.png">

#### Descobertas de IIT
1. Geiger et al. (2022b) desenvolvem o IIT e usam-no para obter resultados SOTA na tarefa MNIST Pointer Value Retrieval (MNIST-PVR; Zhang et al. 2021) e no benchmark de grounded language understanding ReaSCAN (Wu et al. 2021) .
2. Wu et al. (2022b) complementam os objetivos de destilação padrão (Sanh et al. 2019) com um objetivo IIT e mostram que ele melhora em relação às técnicas de destilação padrão.
3. Huang et al. (2022) usam IIT para induzir representações internas de caracteres em LMs com base na tokenização de subpalavras e mostram que isso ajuda em uma variedade de jogos e tarefas em nível de caractere.
4. Wu et al. (2022a) usam o IIT para criar métodos de nível conceitual para explicar o comportamento do modelo.
5. Nosso caderno do curso cobre IIT, bem como abstração causal: ```https://github.com/cgpotts/cs224u/blob/main/iit_equality.ipynb```


### Distributed Alignment Search (DAS)
Introduz o método de busca de alinhamento distribuído (DAS), que pode ser usado para avaliar como os neurônios ou unidades dentro do modelo estão alinhados em relação às diferentes características linguísticas. Isso pode oferecer insights sobre como o modelo representa e processa informações linguísticas.



# NLP Methods and Metrics
## Overview
## Classifier Metrics
## Generation Metrics
## Datasets
## Data Organization
## Model Evaluation




