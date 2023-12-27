# Processamento Neural de Linguagem Natural em Português

# Summary
1. [Semana 1](#semana-1)
   - [Introdução ao Processamento de Linguagem Natural](#introdução-ao-processamento-de-linguagem-natural)
   - [Processamento Baseado em Regras](#processamento-baseado-em-regras)
   - [Modelos Probabilísticos de Linguagem](#modelos-probabilísticos-de-linguagem)
   - [Problemas Típicos em PLN: Problemas Básicos e de Morfosintáxe](#problemas-típicos-em-pln-problemas-básicos-e-de-morfosintáxe)
   - [Problemas Típicos em PLN: Sintaxe, Semântica e Pragmática](#problemas-típicos-em-pln-sintaxe-semântica-e-pragmática)
   - [Lista 01 (Colab)](#lista-01-colab)

2. [Semana 2](#semana-2)
   - [Introdução às Redes Neurais](#introdução-às-redes-neurais)
   - [Perceptron Probabilísticos](#perceptron-probabilísticos)
   - [Redes Multicamadas](#redes-multicamadas)
   - [Treinamento de Redes Neurais](#treinamento-de-redes-neurais)
   - [Lista 02 (Colab)](#lista-02-colab)

3. [Semana 3](#semana-3)
   - [Representação de Palavras: Modelo one-hot](#representação-de-palavras-modelo-one-hot)
   - [Representações Alternativas](#representações-alternativas)
   - [Representação Vetorial de Palavras](#representação-vetorial-de-palavras)
   - [Lista 03 (Colab)](#lista-03-colab)

4. [Semana 4](#semana-4)
   - [Modelo Word2vec Básico de Embedding](#modelo-word2vec-básico-de-embedding)
   - [Detalhamento do Modelo Básico](#detalhamento-do-modelo-básico)
   - [O Modelo Completo](#o-modelo-completo)
   - [Avaliação do Modelo word2vec](#avaliação-do-modelo-word2vec)
   - [Lista 04 (Colab)](#lista-04-colab)
   - [Entrega 01 - Implementando word2vec](#entrega-01-implementando-word2vec)

5. [Semana 5](#semana-5)
   - [Recorrencia Neural](#recorrencia-neural)
   - [Treinamento Recorrente](#treinamento-recorrente)
   - [Modelo Sequência para Sequência](#modelo-sequência-para-sequência)
   - [Problemas de Recorrência e Redes Profundas](#problemas-de-recorrência-e-redes-profundas)
   - [Lista 05 (Colab)](#lista-05-colab)

6. [Semana 6](#semana-6)
   - [Redes LSTM](#redes-lstm)
   - [Redes Recorrentes GRU e Redes Recorrentes Bidirecionais](#redes-recorrentes-gru-e-redes-recorrentes-bidirecionais)
   - [Lista 06 (Colab)](#lista-06-colab)
   - [Entrega 02 - RNNs Bidirecionais LSTM e GRU](#entrega-02-rnns-bidirecionais-lstm-e-gru)

## Semana 1
### Introdução ao Processamento de Linguagem Natural
**Elementos da língua** e áreas que as estudam:
1. **Sons (Acústica):**
   - **Descrição:** Esta área lida com os sons produzidos na fala humana e como são percebidos.
   - **Exemplo:** O som "p" em "palavra" é produzido fechando os lábios e liberando o ar.

2. **Ritmos (Prosódia - Fonologia):**
   - **Descrição:** Prosódia refere-se a elementos como entonação, ritmo e padrões de ênfase na fala.
   - **Exemplo:** A entonação ascendente no final de uma pergunta em português.

3. **Fonema (Fonética):**
   - **Descrição:** Fonemas são os sons distintivos que podem distinguir palavras em uma língua.
   - **Exemplo:** Em português, a diferença entre "pato" e "fato" está no fonema inicial.

4. **Palavras (Morfologia):**
   - **Descrição:** Morfologia estuda a estrutura interna das palavras e como elas são formadas.
   - **Exemplo:** Em "infelizmente", "infeliz" é uma raiz morfológica, e "mente" é um sufixo.

5. **Sintagma (Frases - Sintaxe):**
   - **Descrição:** Sintaxe analisa a estrutura gramatical das frases e como as palavras se combinam.
   - **Exemplo:** "O gato preto" é um sintagma nominal, onde "o gato" é o núcleo.

6. **Significados (Semântica):**
   - **Descrição:** Semântica explora o significado das palavras e como as palavras se relacionam umas com as outras.
   - **Exemplo:** A diferença de significado entre "rápido" e "veloz".

7. **Uso (Pragmática):**
   - **Descrição:** Pragmática estuda como o contexto afeta a interpretação do significado.
   - **Exemplo:** O significado de "pode me passar o sal?" pode variar com base no contexto e na entonação.

Essas áreas representam diferentes aspectos do estudo da linguagem, fornecendo uma compreensão abrangente dos elementos que compõem a comunicação linguística.

### Processamento Baseado em Regras
O Processamento de Linguagem Natural (PLN) baseado em regras gramaticais envolve o uso de um conjunto predefinido de regras linguísticas para analisar e compreender a linguagem. Ao contrário de abordagens mais modernas baseadas em aprendizado de máquina, o PLN baseado em regras depende de regras gramaticais explicitamente definidas para realizar tarefas específicas. Aqui está uma explicação mais detalhada, juntamente com exemplos:

1. **Explicação:**
   - Nesse método, especialistas em linguística ou programadores definem regras gramaticais que descrevem a estrutura e as relações entre as palavras em uma língua.
   - Essas regras são usadas para analisar a sintaxe e a semântica de frases, identificar partes da fala e realizar outras tarefas de processamento de linguagem.

2. **Descrição:**
   - As regras gramaticais geralmente incluem padrões sintáticos, estruturas de frase, relações gramaticais e significados associados a determinadas construções linguísticas.
   - A análise é realizada seguindo rigorosamente essas regras, muitas vezes usando uma abordagem de análise sintática.

3. **Exemplos:**
   - *Análise Sintática:* Uma regra gramatical pode ser definida para identificar sujeito, verbo e objeto em uma frase. Por exemplo, a regra pode indicar que um sujeito é geralmente seguido por um verbo e um objeto, e isso pode ser usado para analisar a estrutura da frase.
   
   - *Extração de Informações:* Se uma regra gramatical estabelece que certos padrões de palavras indicam a presença de informações específicas (por exemplo, "Nome: [nome]" ou "Data de nascimento: [data]"), o sistema pode seguir essas regras para extrair essas informações de um texto.

   - *Tradução:* Em sistemas de tradução automática baseados em regras, regras gramaticais são usadas para mapear estruturas gramaticais de uma língua para outra, seguindo padrões estabelecidos.

Embora o PLN baseado em regras tenha sido uma abordagem inicial no desenvolvimento de sistemas de processamento de linguagem, ela tem limitações, especialmente em lidar com ambiguidades e variações na linguagem. Muitas abordagens modernas incorporam técnicas de aprendizado de máquina para lidar com essas complexidades.

**Modelagem Matemática da Linguagem: Modelos Simbólicos, Não-Numéricos ou Qualitativos:**

1. **Morfologia e Sintaxe: Linguagens Formais e Autômatos:**
   - **Explicação:** A modelagem matemática de morfologia e sintaxe envolve o uso de linguagens formais para descrever a estrutura das palavras (morfologia) e a composição das frases (sintaxe).
   - **Exemplo:** As gramáticas formais, como as gramáticas livres de contexto, podem ser usadas para descrever a estrutura sintática de uma língua. Autômatos, como autômatos de pilha, podem ser aplicados para reconhecer padrões morfológicos.

2. **Sintaxe: Gramáticas Formais e Lógica Categórica:**
   - **Explicação:** A sintaxe é modelada usando gramáticas formais que definem as regras de formação de frases. A lógica categórica é uma abordagem que utiliza categorias gramaticais para representar a estrutura sintática.
   - **Exemplo:** Uma gramática formal pode descrever as regras para criar uma frase em inglês, enquanto a lógica categórica pode representar a estrutura sintática usando categorias como sujeito, verbo e objeto.

3. **Semântica: Lógica Categórica:**
   - **Explicação:** A semântica é modelada usando lógica categórica para representar o significado das expressões linguísticas. Ela estabelece relações entre as categorias gramaticais e seus significados.
   - **Exemplo:** A lógica categórica pode ser usada para representar como as palavras e frases se relacionam semanticamente, capturando o significado e as relações entre conceitos.

Essas abordagens matemáticas fornecem estruturas formais para entender e representar a linguagem. Elas são fundamentais em disciplinas como a teoria da computação, linguística computacional e processamento de linguagem natural, onde a formalização da linguagem permite a criação de modelos precisos e eficientes para análise e processamento automático.

As gramáticas livres de contexto são uma classe específica de gramáticas formais utilizadas para descrever a estrutura sintática de linguagens formais. Elas desempenham um papel fundamental na teoria da computação, linguística computacional e no desenvolvimento de compiladores.

**Gramática livre de contexto**

A definição formal de uma gramática livre de contexto inclui quatro componentes principais:

1. **Conjunto de Símbolos Terminais (T):**
   - São os símbolos que aparecem nas cadeias finais derivadas pela gramática. Em uma gramática para a linguagem de programação, por exemplo, os símbolos terminais podem representar palavras-chave, operadores e identificadores.

2. **Conjunto de Símbolos Não-Terminais (N):**
   - São símbolos que podem ser substituídos por sequências de símbolos terminais e/ou não-terminais. Eles são utilizados como variáveis na definição de regras gramaticais.

3. **Conjunto de Regras de Produção (P):**
   - São regras que especificam como os símbolos não-terminais podem ser substituídos por sequências de símbolos terminais e/ou não-terminais. Cada regra geralmente assume a forma "A -> β", onde "A" é um símbolo não-terminal e "β" é uma sequência de símbolos terminais e/ou não-terminais.

4. **Símbolo Inicial (S):**
   - É o símbolo não-terminal a partir do qual as derivações começam. Geralmente, é um único símbolo não-terminal.

**Exemplo de Gramática Livre de Contexto:**
Considerando uma gramática para representar expressões aritméticas simples:

- Símbolos Terminais (T): {+, -, *, /, números}
- Símbolos Não-Terminais (N): {Expr, Term, Factor}
- Regras de Produção (P):
  1. Expr -> Expr + Term | Expr - Term | Term
  2. Term -> Term * Factor | Term / Factor | Factor
  3. Factor -> (Expr) | números

Neste exemplo, "Expr" representa uma expressão aritmética, "Term" um termo e "Factor" um fator. As regras especificam como esses elementos podem ser combinados para formar expressões aritméticas válidas.

Gramáticas livres de contexto são fundamentais em compiladores para análise sintática, na representação de linguagens de programação e no desenvolvimento de parsers para processamento de linguagem natural. Elas possuem uma estrutura poderosa e são amplamente utilizadas em diversos campos da ciência da computação.

**Gramática e Estrutura da Frase**

A gramática e a estrutura da frase são elementos essenciais na compreensão da linguagem. Vamos explorar esses conceitos, fornecendo uma explicação geral e exemplos:

**Gramática:**
A gramática é o conjunto de regras que governa a estrutura e o uso de uma língua. Ela inclui regras sintáticas (estrutura de frases), regras morfológicas (formação de palavras) e regras semânticas (significado das palavras e frases).

**Estrutura da Frase:**
A estrutura da frase refere-se à organização gramatical das palavras em uma sentença para expressar uma ideia completa. Envolve a disposição de elementos como sujeito, verbo, objeto, adjetivos e advérbios de maneira coerente.

**Exemplos:**
Vamos analisar a estrutura de uma frase simples:

1. **Frase Simples:**
   - **Exemplo:** "O gato caçou um rato."
   - **Estrutura:**
     - Sujeito: "O gato"
     - Verbo: "caçou"
     - Objeto: "um rato"

2. **Frase com Modificadores:**
   - **Exemplo:** "O rápido e ágil cão perseguia a bola incansavelmente."
   - **Estrutura:**
     - Sujeito: "O rápido e ágil cão"
     - Verbo: "perseguia"
     - Objeto: "a bola"
     - Advérbio: "incansavelmente"

3. **Frase com Cláusulas:**
   - **Exemplo:** "Quando o sol se pôs, as estrelas começaram a aparecer."
   - **Estrutura:**
     - Cláusula Temporal: "Quando o sol se pôs"
     - Sujeito: "as estrelas"
     - Verbo: "começaram a aparecer"

4. **Frase Interrogativa:**
   - **Exemplo:** "Você já almoçou?"
   - **Estrutura:**
     - Pronome Interrogativo: "Você"
     - Verbo: "almoçou"

Estes exemplos ilustram diferentes aspectos da estrutura da frase, incluindo a presença de sujeito, verbo, objeto, modificadores e a formação de frases interrogativas. A compreensão da gramática e da estrutura da frase é fundamental para a comunicação eficaz e a interpretação correta da linguagem.

**Categorias Morfosintáticas**

As categorias morfosintáticas referem-se às classes gramaticais ou categorias sintáticas que as palavras de uma língua podem ocupar em uma frase. Elas combinam elementos morfológicos (relativos à forma ou estrutura da palavra) e elementos sintáticos (relativos à função da palavra em uma sentença). As principais categorias morfosintáticas incluem:

1. **Substantivo (N):**
   - **Exemplo:** "casa," "gato," "felicidade"
   - **Função:** Representa pessoas, lugares, objetos ou ideias.

2. **Verbo (V):**
   - **Exemplo:** "correr," "pular," "cantar"
   - **Função:** Expressa ação, estado ou processo.

3. **Adjetivo (Adj):**
   - **Exemplo:** "bonito," "rápido," "inteligente"
   - **Função:** Modifica substantivos, fornecendo características ou qualidades.

4. **Advérbio (Adv):**
   - **Exemplo:** "rapidamente," "bem," "hoje"
   - **Função:** Modifica verbos, adjetivos ou outros advérbios, indicando circunstâncias.

5. **Pronome (Pron):**
   - **Exemplo:** "eu," "ela," "eles"
   - **Função:** Substitui ou faz referência a um substantivo.

6. **Preposição (Prep):**
   - **Exemplo:** "em," "sobre," "com"
   - **Função:** Estabelece relações espaciais, temporais ou lógicas entre palavras na frase.

7. **Conjunção (Conj):**
   - **Exemplo:** "e," "mas," "ou"
   - **Função:** Conecta palavras, frases ou orações.

8. **Artigo (Art):**
   - **Exemplo:** "o," "uma," "os"
   - **Função:** Indica se um substantivo é específico ou genérico.

Essas categorias são fundamentais para a análise sintática e semântica de uma sentença. Além disso, elas contribuem para a estruturação e compreensão da língua. Em muitas línguas, as palavras podem pertencer a diferentes categorias, dependendo do contexto ou da função que desempenham em uma frase específica. A categorização morfosintática é uma ferramenta crucial para a descrição e análise linguística.

**Regras Lexicais e Regras Gramaticais**

**Regras Lexicais:**

As regras lexicais lidam com a estrutura e a formação de palavras, incluindo morfemas (unidades mínimas de significado) e a maneira como as palavras são formadas e modificadas.

1. **Exemplo:**
   - Na língua inglesa, a adição do sufixo "-ly" a um adjetivo forma um advérbio. Exemplo: "quick" (adjetivo) -> "quickly" (advérbio).

2. **Função:**
   - Estabelecem padrões para a construção e derivação de palavras, determinando como radicais, prefixos e sufixos podem ser combinados.

**Regras Gramaticais:**

As regras gramaticais governam a estrutura e a organização de frases, especificando como as palavras devem ser combinadas para formar unidades significativas.

1. **Exemplo:**
   - Em uma gramática simples, a regra para uma frase pode ser expressa como "Sujeito + Verbo + Objeto". Exemplo: "O gato (Sujeito) caçou (Verbo) um rato (Objeto)."

2. **Função:**
   - Determinam a ordem e a relação entre os constituintes de uma sentença, garantindo que a construção seja gramaticalmente correta.

**Comparação:**

1. **Regras Lexicais vs. Regras Gramaticais:**
   - **Regras Lexicais:** Focam na formação e na estruturação de palavras.
   - **Regras Gramaticais:** Concentram-se na organização e na estrutura de frases.

2. **Exemplo Conjunto:**
   - Considere a frase "O estudante aprendeu rapidamente."
   - **Regra Lexical:** A formação do advérbio "rapidamente" a partir do adjetivo "rápido".
   - **Regra Gramatical:** A estrutura da frase, como "Sujeito + Verbo + Advérbio."

3. **Importância Conjunta:**
   - Ambas as regras são essenciais para a compreensão e a produção linguística. As regras lexicais contribuem para o vocabulário e a expressividade, enquanto as regras gramaticais garantem a coerência e a clareza na comunicação.

Em resumo, as regras lexicais e gramaticais trabalham juntas para formar uma base sólida na compreensão e na produção da linguagem, abrangendo desde a construção de palavras até a organização de frases em contextos gramaticalmente corretos.

**Gramática Livre de Contexto (GLC):**

1. **Explicação:**
   - Uma Gramática Livre de Contexto é um tipo específico de gramática formal que descreve a estrutura sintática de uma linguagem. Ela consiste em um conjunto de regras de produção que especificam como as cadeias de símbolos podem ser formadas.

2. **Exemplo de Regra de Produção:**
   - Considere a regra para uma frase simples em inglês: `S -> NP VP`. Isso significa que uma sentença (S) pode ser formada por um sintagma nominal (NP) seguido por um sintagma verbal (VP).

**Derivação Sintática:**

1. **Explicação:**
   - A derivação sintática é o processo de aplicar as regras de produção de uma gramática para gerar uma sequência de símbolos que forma uma sentença na linguagem.

2. **Exemplo de Derivação:**
   - Dada a regra `S -> NP VP`, poderíamos derivar a sentença "O gato caçou" da seguinte maneira:
     1. `S` (inicial)
     2. `NP VP` (aplicando a regra `S -> NP VP`)
     3. `Det N VP` (selecionando "O" para NP)
     4. `O N VP` (selecionando "gato" para N)
     5. `O gato VP` (aplicando regras adicionais para VP)
     6. `O gato V` (selecionando "caçou" para V)
     7. `O gato caçou` (aplicando regras adicionais se necessário)

**Árvore de Derivação Sintática:**

1. **Explicação:**
   - Uma árvore de derivação sintática é uma representação gráfica da derivação sintática, mostrando como as regras de produção são aplicadas para gerar a sentença.

2. **Exemplo de Árvore de Derivação:**
   - Usando o exemplo anterior, a árvore de derivação pode ser representada assim:

```
       S
      / \
     NP  VP
    /  |   \
 Det   N    V
  |    |    |
  O  gato caçou
```

Nesta árvore, cada nó representa um símbolo da gramática e as arestas indicam a aplicação de uma regra de produção. A raiz da árvore é o símbolo inicial (`S`), e as folhas representam os símbolos terminais que formam a sentença final.

Esses conceitos são fundamentais na análise sintática de linguagens formais e são amplamente utilizados em linguística computacional, processamento de linguagem natural e design de compiladores. A Gramática Livre de Contexto, a derivação sintática e as árvores de derivação são ferramentas poderosas para descrever e analisar a estrutura sintática das linguagens.

**Gramática Ambígua:**
Uma gramática é considerada ambígua quando uma determinada sentença pode ser analisada de maneiras diferentes, resultando em mais de uma árvore sintática possível para a mesma sequência de palavras. A ambiguidade pode surgir quando as regras gramaticais permitem mais de uma interpretação válida.

**Exemplo: "Eu vi o menino com o telescópio":**
   - Nesta sentença, a ambiguidade surge devido à possível interpretação dupla da preposição "com". A frase pode ser entendida como "Eu vi o menino usando o telescópio" ou "Eu vi o menino que tinha o telescópio".

**Árvores Sintáticas Ambíguas:**

1. **Primeira Interpretação (usando o telescópio):**
   
   ```
         S
        / \
      NP   VP
      |   / | \
      N  V  Det N
      |   |    |
     Eu  vi  o menino
                  |
                  P
                  |
                 com
                  |
              Det   N
              |     |
              o  telescópio
   ```

2. **Segunda Interpretação (com o telescópio):**
   
   ```
         S
        / \
      NP   VP
      |   / | \
      N  V  Det N
      |   |    |
     Eu  vi  o menino
                  |
                  P
                  |
                 com
                  |
                Det   N
                |     |
                o  telescópio
   ```

Nas duas árvores, a estrutura geral da sentença é a mesma, mas a interpretação do papel da preposição "com" cria ambiguidade. Uma árvore sintática pode representar a interpretação de que o telescópio foi usado pelo observador (primeira árvore), enquanto a outra árvore representa a interpretação de que o telescópio pertence ao menino (segunda árvore).

A ambiguidade em gramáticas pode causar problemas de interpretação e é um desafio para sistemas de processamento de linguagem natural, pois eles precisam ser capazes de escolher a interpretação mais apropriada em contextos específicos.

**A Hierarquia de Chomsky**

A Hierarquia de Chomsky é uma classificação de gramáticas formais proposta pelo linguista Noam Chomsky. Essa hierarquia organiza as gramáticas em quatro níveis, cada um representando um conjunto diferente de linguagens. As quatro classes na Hierarquia de Chomsky, em ordem crescente de complexidade, são:

1. **Tipo 3 - Gramáticas Regulares:**
   - **Descrição:** São gramáticas simples, capazes de gerar linguagens regulares. Elas são menos poderosas em termos de expressividade do que os tipos subsequentes.
   - **Exemplo:** Gramáticas regulares são frequentemente usadas para descrever padrões em expressões regulares, como a linguagem dos números binários.

2. **Tipo 2 - Gramáticas Livres de Contexto (GLC):**
   - **Descrição:** Gramáticas que geram linguagens livres de contexto. São mais expressivas que as gramáticas regulares e são amplamente usadas em linguagens de programação e análise sintática.
   - **Exemplo:** A gramática que gera a linguagem das expressões aritméticas, como `E -> E + E`, pertence a esta categoria.

3. **Tipo 1 - Gramáticas Sensíveis ao Contexto:**
   - **Descrição:** Gramáticas mais poderosas que as gramáticas livres de contexto, permitindo regras que levam em consideração o contexto em que uma produção ocorre.
   - **Exemplo:** Gramáticas sensíveis ao contexto são usadas em linguagens naturais, onde o significado de uma palavra pode depender do contexto da sentença.

4. **Tipo 0 - Gramáticas Irrestritas:**
   - **Descrição:** Gramáticas sem restrições, permitindo regras arbitrárias. São a classe mais poderosa, mas também a mais difícil de analisar e processar.
   - **Exemplo:** Linguagens definidas por máquinas de Turing são um exemplo de linguagens geradas por gramáticas irrestritas.

Essa hierarquia é significativa porque mostra a relação entre diferentes tipos de gramáticas e as classes de linguagens que elas podem gerar. À medida que se move de um tipo para outro, as gramáticas tornam-se mais expressivas, mas também mais difíceis de analisar e processar automaticamente. A Hierarquia de Chomsky é fundamental no estudo da teoria da computação e fornece uma estrutura para entender a complexidade das linguagens formais.





### Modelos Probabilísticos de Linguagem
### Problemas Típicos em PLN: Problemas Básicos e de Morfosintáxe
### Problemas Típicos em PLN: Sintaxe, Semântica e Pragmática
### Lista 01 (Colab)

## Semana 2
### Introdução às Redes Neurais
### Perceptron Probabilísticos
### Redes Multicamadas
### Treinamento de Redes Neurais
### Lista 02 (Colab)

## Semana 3
### Representação de Palavras: Modelo one-hot
### Representações Alternativas
### Representação Vetorial de Palavras
### Lista 03 (Colab)

## Semana 4
### Modelo Word2vec Básico de Embedding
### Detalhamento do Modelo Básico
### O Modelo Completo
### Avaliação do Modelo word2vec
### Lista 04 (Colab)
### Entrega 01 - Implementando word2vec

## Semana 5
### Recorrencia Neural
### Treinamento Recorrente
### Modelo Sequência para Sequência
### Problemas de Recorrência e Redes Profundas
### Lista 05 (Colab)

## Semana 6
### Redes LSTM
### Redes Recorrentes GRU e Redes Recorrentes Bidirecionais
### Lista 06 (Colab)
### Entrega 02 - RNNs Bidirecionais LSTM e GRU
