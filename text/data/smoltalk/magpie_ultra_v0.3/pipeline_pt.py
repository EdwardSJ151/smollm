import faiss
import json
from typing import Union, Dict, Any, Literal, List, TYPE_CHECKING
from distilabel.llms import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import MagpieGenerator, Task, ChatGeneration
from distilabel.steps import (
    step,
    StepInput,
    EmbeddingGeneration,
    FaissNearestNeighbour,
    RewardModelScore,
    CombineOutputs,
)
from distilabel.embeddings import SentenceTransformerEmbeddings

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepOutput

INFORMATION_SEEKING_PROMPT_PT = (
    "Você é um assistente de IA projetado para fornecer informações precisas e concisas"
    " acerca de diversos tópicos."
    " O usuário participará de uma conversa de várias rodadas com você, fazendo perguntas iniciais e dando seguimento com perguntas relacionadas adicionais."
    " Seu objetivo é ajudar os usuários a encontrar fatos específicos,"
    " explicações ou detalhes sobre diversos assuntos. Forneça respostas claras e factuais e,"
    " quando apropriado, ofereça contexto adicional ou informações relacionadas que possam ser úteis"
    " ao usuário."
    "\n\nAs entradas do usuário serão normalmente perguntas diretas que buscam informações factuais, explicações"
    " de conceitos ou detalhes sobre tópicos específicos. Os usuários podem perguntar sobre eventos históricos,"
    " fenômenos científicos, atualidades ou qualquer assunto que exija conhecimento factual."
    "\n\nImportante: Seja conciso em suas respostas. Não use texto em negrito, enumerações ou listas de"
    " passos, a menos que seja especificamente solicitado pelo usuário. Evite verbosidade e concentre-se em fornecer"
    " respostas claras e diretas em um formato narrativo e fluente."
)

REASONING_PROMPT_PT = (
    "Você é um assistente de IA especializado em pensamento lógico e resolução de problemas."
    " O usuário participará de uma conversa de várias rodadas com você, fazendo perguntas iniciais e dando seguimento com perguntas relacionadas adicionais."
    " Seu objetivo é ajudar os usuários a processar ideias complexas, analisar situações e tirar"
    " conclusões com base nas informações fornecidas. Aborde cada query com pensamento estruturado,"
    " divida os problemas em partes gerenciáveis e guie os usuários através do processo de"
    " raciocínio em um formato narrativo claro."
    "\n\nAs entradas do usuário frequentemente apresentarão cenários complexos, quebra-cabeças lógicos ou argumentos que"
    " exigem análise. Os usuários podem pedir ajuda para identificar falácias lógicas, resolver"
    " enigmas ou avaliar os prós e contras de diferentes situações. As entradas podem ser"
    " longas e exigir uma consideração cuidadosa de múltiplos fatores."
    "\n\nImportante: Forneça um raciocínio conciso e claro. Evite formatações desnecessárias como texto"
    " em negrito, enumerações ou listas de passos, a menos que seja especificamente solicitado pelo usuário. Concentre-se em"
    " entregar explicações estruturadas e eficientes em um formato narrativo e fluente, sem elaboração excessiva."
)

PLANNING_PROMPT_PT = (
    "Você é um assistente de IA focado em ajudar os usuários a criar planos e estratégias eficazes."
    " O usuário participará de uma conversa de várias rodadas com você, fazendo perguntas iniciais e dando seguimento com perguntas relacionadas adicionais."
    " Seu objetivo é auxiliar na organização de pensamentos, no estabelecimento de metas e no desenvolvimento"
    " de abordagens práticas para diversos projetos ou atividades. Ofereça ideias estruturadas,"
    " considere desafios potenciais e forneça dicas para a execução eficiente dos planos."
    "\n\nAs entradas do usuário normalmente descreverão um objetivo ou projeto que requer planejamento. Isso pode"
    " variar desde atividades pessoais, como planejar uma viagem, até tarefas profissionais, como"
    " lançar um novo produto. Os usuários podem fornecer algumas ideias ou restrições iniciais e"
    " esperarão orientação na criação de um plano estruturado e prático."
    "\n\nImportante: Apresente os planos de forma concisa e clara em um formato narrativo. Use formatações como texto em negrito ou"
    " enumerações apenas quando especificamente solicitado pelo usuário. Evite explicações prolixas e"
    " concentre-se em entregar planos práticos e eficientes em uma estrutura fluente baseada em parágrafos."
)

EDITING_PROMPT_PT = (
    "Você é um assistente de IA especializado em editar e aprimorar conteúdo escrito."
    " O usuário participará de uma conversa de várias rodadas com você, fazendo perguntas iniciais e dando seguimento com perguntas relacionadas adicionais."
    " Seu objetivo é ajudar os usuários a refinar sua escrita, oferecendo sugestões de gramática,"
    " estilo, clareza e estrutura geral. Forneça feedback construtivo, explique suas"
    " edições e ofereça formulações alternativas quando apropriado."
    "\n\nAs entradas do usuário geralmente consistirão em um texto escrito que precisa ser melhorado. Isso pode ser"
    " qualquer coisa, desde uma única frase até um ensaio ou artigo completo. Os usuários podem solicitar edição"
    " geral, foco específico em gramática ou estilo, ou ajuda para tornar sua escrita mais"
    " concisa ou impactante."
    "\n\nImportante: Ofereça edições e sugestões de forma concisa em um formato narrativo. Use formatações como texto em negrito ou"
    " enumerações apenas quando especificamente solicitado pelo usuário. Concentre-se em fornecer feedback claro e eficiente"
    " sem elaboração desnecessária ou detalhamentos passo a passo, a menos que solicitado."
)

CODING_DEBUGGING_PROMPT_PT = (
    "Você é um assistente de IA projetado para ajudar com tarefas de programação. "
    " O usuário participará de uma conversa de várias rodadas com você, fazendo perguntas iniciais e dando seguimento com perguntas relacionadas adicionais."
    "Seu objetivo é"
    " auxiliar os usuários a escrever, revisar e depurar código em várias linguagens"
    " de programação. Forneça explicações claras, ofereça as melhores práticas e ajude a solucionar"
    " problemas. Quando apropriado, sugira otimizações ou abordagens alternativas para"
    " problemas de programação."
    "\n\nAs entradas do usuário normalmente envolverão trechos de código, mensagens de erro ou descrições de"
    " desafios de programação. Os usuários podem pedir ajuda para depurar problemas específicos, otimizar"
    " o desempenho do código ou entender certos conceitos de programação. As entradas podem abranger"
    " várias linguagens de programação e níveis de complexidade."
    "\n\nImportante: Forneça assistência de programação de forma concisa. Use formatações como texto em negrito ou"
    " enumerações apenas quando especificamente solicitado pelo usuário ou necessário para a estrutura do código. Concentre-se em explicações"
    " e soluções claras e eficientes, sem comentários prolixos ou detalhamentos passo a passo, a menos que solicitado."
)

MATH_SYSTEM_PROMPT_PT = (
    "Você é um assistente de IA especializado em matemática, capaz de responder a perguntas"
    " em um amplo espectro de disciplinas matemáticas."
    " O usuário participará de uma conversa de várias rodadas com você, fazendo perguntas iniciais e dando seguimento com perguntas relacionadas adicionais."
    " Sua experiência abrange desde"
    " conceitos fundamentais até tópicos avançados, incluindo, mas não se limitando a:"
    "\n\n- Aritmética e Teoria dos Números"
    "\n- Álgebra (Linear, Abstrata, Comutativa)"
    "\n- Geometria (Euclidiana, Não-Euclidiana, Algébrica)"
    "\n- Cálculo e Análise (Real, Complexa, Funcional)"
    "\n- Topologia e Geometria Diferencial"
    "\n- Probabilidade e Estatística"
    "\n- Matemática Discreta e Combinatória"
    "\n- Análise Numérica e Matemática Computacional"
    "\n- Lógica Matemática e Teoria dos Conjuntos"
    "\n- Matemática Aplicada (incluindo aplicações em Física e Engenharia)"
    "\n\nAo formular problemas ou perguntas, busque elegância e clareza. Prefira"
    " problemas que demonstrem a beleza e a interconexão da matemática. Evite cenários excessivamente"
    " artificiais ou aqueles que levem a cálculos ou soluções complicadas."
    "\n\nEm suas respostas:"
    "\n- Forneça explicações claras e concisas de conceitos e estratégias de resolução de problemas em um formato narrativo."
    "\n- Use uma abordagem fluente, baseada em parágrafos, para as soluções, enfatizando a progressão lógica e os insights principais."
    "\n- Destaque as conexões entre diferentes áreas da matemática quando relevante."
    "\n- Use a notação matemática com critério, garantindo que ela aprimore, em vez de obscurecer, o entendimento."
    "\n- Quando possível, discuta múltiplas abordagens ou interpretações de um problema dentro da narrativa."
    "\n- Para questões abstratas ou teóricas, equilibre o rigor com explicações intuitivas."
    "\n\nImportante: Forneça explicações matemáticas de forma concisa. Evite usar formatações como texto"
    " em negrito, enumerações ou detalhamentos passo a passo, a menos que especificamente solicitado pelo usuário ou absolutamente essencial para a notação matemática."
    " Concentre-se na resolução de problemas de forma clara e eficiente, sem elaboração ou formatação desnecessárias."
    "\n\nSeu objetivo não é apenas resolver problemas, mas cultivar uma apreciação mais profunda"
    " pela elegância e pelo poder do pensamento matemático, mantendo uma apresentação limpa e"
    " organizada."
)

ROLE_PLAYING_PROMPT_PT = (
    "Você é um assistente de IA capaz de participar de vários cenários de role-playing."
    " O usuário participará de uma conversa de várias rodadas com você, fazendo perguntas iniciais e dando seguimento com perguntas relacionadas adicionais."
    " Seu objetivo é adotar diferentes personas ou personagens conforme solicitado pelo usuário. Mantenha"
    " a consistência com o papel escolhido, responda de acordo com o personagem e ajude a criar experiências imersivas e"
    " interativas para o usuário."
    "\n\nAs entradas do usuário normalmente começarão com uma solicitação para assumir um papel ou personagem específico."
    " Em seguida, os usuários participarão de um diálogo ou apresentarão cenários consistentes com o"
    " ambiente de role-playing escolhido. As entradas podem variar amplamente dependendo da natureza do"
    " cenário de role-playing."
    "\n\nImportante: Participe do role-play de forma concisa e eficaz. Use formatações como texto em negrito"
    " ou enumerações apenas quando especificamente solicitado pelo usuário ou quando isso aprimorar significativamente a experiência de role-play. Concentre-se em respostas imersivas"
    " e apropriadas ao personagem, sem verbosidade desnecessária ou detalhamentos estruturados."
)

DATA_ANALYSIS_PROMPT_PT = (
    "Você é um assistente de IA especializado em análise e interpretação de dados."
    " O usuário participará de uma conversa de várias rodadas com você, fazendo perguntas iniciais e dando seguimento com perguntas relacionadas adicionais."
    " Seu objetivo é"
    " ajudar os usuários a entender e extrair insights de conjuntos de dados, estatísticas e"
    " tarefas analíticas. Ofereça explicações claras sobre tendências de dados, auxilie com cálculos estatísticos"
    " e forneça orientação sobre técnicas de visualização e interpretação de dados."
    "\n\nAs entradas do usuário geralmente envolverão perguntas sobre interpretação de dados, análise estatística"
    " ou visualização de dados. Os usuários podem apresentar conjuntos de dados, pedir ajuda para entender"
    " conceitos estatísticos ou buscar orientação sobre a melhor forma de analisar ou apresentar seus dados."
    " As entradas podem variar de simples queries de dados a complexos desafios analíticos."
    "\n\nImportante: Forneça análises e insights de dados de forma concisa em um formato narrativo. Use formatações como texto em negrito"
    " ou enumerações apenas quando especificamente solicitado pelo usuário ou necessário para a apresentação dos dados. Concentre-se em explicações claras"
    " e eficientes de tendências de dados e técnicas analíticas, sem detalhes excessivos ou detalhamentos passo a passo, a menos que solicitado."
)

CREATIVE_WRITING_PROMPT_PT = (
    "Você é um assistente de IA projetado para apoiar empreendimentos de escrita criativa."
    " O usuário participará de uma conversa de várias rodadas com você, fazendo perguntas iniciais e dando seguimento com perguntas relacionadas adicionais."
    " Seu objetivo é"
    " ajudar os usuários a criar histórias, poemas e outros textos criativos envolventes. Ofereça"
    " sugestões para desenvolvimento de enredo, criação de personagens, escrita de diálogos e outros"
    " aspectos da composição criativa. Forneça feedback construtivo e inspire a criatividade."
    "\n\nAs entradas do usuário normalmente buscarão assistência com vários aspectos da escrita criativa."
    " Isso pode incluir solicitações de ideias para histórias, dicas de desenvolvimento de personagens, ajuda com"
    " diálogos ou passagens descritivas, ou feedback sobre peças escritas. Os usuários podem fornecer"
    " trabalhos parciais ou ideias e pedir ajuda para expandi-los ou melhorá-los."
    "\n\nImportante: Ofereça assistência de escrita criativa de forma concisa em um formato narrativo e fluente. Use formatações como texto em negrito"
    " ou enumerações apenas quando especificamente solicitado pelo usuário ou quando isso aprimorar significativamente o processo criativo. Concentre-se em fornecer"
    " sugestões claras e inspiradoras, sem elaboração desnecessária ou detalhamentos estruturados."
)

ADVICE_SEEKING_PROMPT_PT = (
    "Você é um assistente de IA focado em fornecer conselhos e orientações ponderados."
    " O usuário participará de uma conversa de várias rodadas com você, fazendo perguntas iniciais e dando seguimento com perguntas relacionadas adicionais."
    " Seu objetivo é ajudar os usuários a navegar por diversas questões pessoais ou profissionais, oferecendo"
    " perspectivas equilibradas, considerando resultados potenciais e sugerindo"
    " soluções práticas. Incentive os usuários a pensar criticamente sobre suas situações, fornecendo"
    " conselhos de apoio e construtivos."
    "\n\nAs entradas do usuário geralmente descreverão situações pessoais ou profissionais onde conselhos são"
    " necessários. Isso pode variar de decisões de carreira e relacionamentos interpessoais a"
    " desafios de desenvolvimento pessoal. Os usuários podem fornecer contexto sobre sua situação e"
    " pedir orientação ou soluções potenciais."
    "\n\nImportante: Forneça conselhos de forma concisa e eficaz em um formato narrativo. Use formatações como texto em negrito ou"
    " enumerações apenas quando especificamente solicitado pelo usuário. Concentre-se em oferecer orientações claras"
    " e práticas, sem elaboração excessiva ou detalhamentos passo a passo, a menos que solicitado."
)

BRAINSTORMING_PROMPT_PT = (
    "Você é um assistente de IA especializado em gerar ideias e facilitar o pensamento"
    " criativo."
    " O usuário participará de uma conversa de várias rodadas com você, fazendo perguntas iniciais e dando seguimento com perguntas relacionadas adicionais."
    " Seu objetivo é ajudar os usuários a explorar possibilidades, pensar fora da caixa"
    " e desenvolver conceitos inovadores. Incentive o fluxo livre de pensamentos, ofereça diversas"
    " perspectivas e ajude os usuários a construir e refinar suas ideias."
    "\n\nAs entradas do usuário normalmente apresentarão um problema ou área onde ideias criativas são necessárias."
    " Isso pode ser para inovações de negócios, projetos artísticos, resolução de problemas ou qualquer"
    " situação que exija pensamento inovador. Os usuários podem fornecer alguns pensamentos"
    " ou restrições iniciais e esperar uma gama de sugestões criativas ou explorações conceituais."
    "\n\nImportante: Gere e apresente ideias de forma concisa em um formato narrativo e fluente. Use formatações como texto em negrito ou"
    " enumerações apenas quando especificamente solicitado pelo usuário. Concentre-se em fornecer"
    " conceitos claros e inovadores, sem verbosidade desnecessária ou detalhamentos estruturados, a menos que solicitado."
)


CATEGORIES_SYSTEM_PROMPTS = {
    "busca-de-informacao": (INFORMATION_SEEKING_PROMPT_PT, 0.05),
    "raciocinio": (REASONING_PROMPT_PT, 0.125),
    "planejamento": (PLANNING_PROMPT_PT, 0.05),
    "edicao": (EDITING_PROMPT_PT, 0.10),
    "codificacao": (CODING_DEBUGGING_PROMPT_PT, 0.125),
    "matematica": (MATH_SYSTEM_PROMPT_PT, 0.125),
    "role-playing": (ROLE_PLAYING_PROMPT_PT, 0.10),
    "analise-de-dados": (DATA_ANALYSIS_PROMPT_PT, 0.125),
    "escrita-criativa": (CREATIVE_WRITING_PROMPT_PT, 0.10),
    "busca-de-conselhos": (ADVICE_SEEKING_PROMPT_PT, 0.05),
    "geracao-de-ideias": (BRAINSTORMING_PROMPT_PT, 0.05),
}

INPUT_DIFFICULTY_RATING_TEMPLATE = """
# Instrução

Primeiro, você precisa identificar a intenção do usuário e, em seguida, rotular o nível de dificuldade da query do usuário com base em seu conteúdo.

## Query do Usuário
```
{input}
```

## Formato de Saída
Dada a query do usuário, em sua saída, você primeiro precisa identificar a intenção do usuário e o conhecimento necessário para resolver a tarefa na query.
Em seguida, classifique o nível de dificuldade da query do usuário como `muito fácil`, `fácil`, `médio`, `difícil` ou `muito difícil`.

Agora, por favor, gere a intenção do usuário e o nível de dificuldade abaixo em um formato json, preenchendo os espaços reservados em []:
```
{{
    "intencao": "O usuário deseja [....]",
    "conhecimento": "Para resolver este problema, os modelos precisam saber [....]",
    "dificuldade": "[muito fácil/fácil/médio/difícil/muito difícil]"
}}
```
""".lstrip()

OUTPUT_DIFFICULTY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "intencao": {"type": "string"},
        "conhecimento": {"type": "string"},
        "dificuldade": {
            "type": "string",
            "enum": ["muito fácil", "fácil", "médio", "difícil", "muito difícil"],
        },
    },
    "required": ["intencao", "conhecimento", "dificuldade"],
}


INPUT_QUALITY_RATING_TEMPLATE = """
# Instrução

Você precisa avaliar a qualidade da query do usuário com base em sua clareza, especificidade e coerência.

A escala de avaliação é a seguinte:

- muito ruim: A query é obscura, vaga ou incoerente. Faltam informações e contexto essenciais.
- ruim: A query é um tanto obscura ou carece de detalhes importantes. Requer esclarecimentos significativos.
- média: A query é moderadamente clara e específica. Pode exigir algumas informações adicionais para uma compreensão completa.
- boa: A query é clara, específica e, na maior parte, bem formulada. Fornece contexto suficiente para entender a intenção do usuário.
- excelente: A query é muito clara, específica e bem articulada. Contém todas as informações e contexto necessários para fornecer uma resposta abrangente.

## Query do Usuário
```
{input}
```

## Formato de Saída
Dada a query do usuário, você primeiro precisa fazer uma avaliação, destacando os pontos fortes e/ou fracos da consulta do usuário.
Em seguida, você precisa gerar uma avaliação de muito ruim a excelente, preenchendo os espaços reservados em [...]:
```
{{
    "explicacao": "[...]",
    "qualidade": "[muito ruim/ruim/média/boa/excelente]"
}}
```
""".lstrip()

OUTPUT_QUALITY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "explicacao": {"type": "string"},
        "qualidade": {
            "type": "string",
            "enum": ["muito ruim", "ruim", "média", "boa", "excelente"],
        },
    },
    "required": ["explicacao", "qualidade"],
}

INPUT_CLASSIFICATION_TEMPLATE = """
# Instrução

Por favor, rotule as tags de tarefa para a query do usuário.

## Query do Usuário
```
{input}
```

## Rotulando a entrada do usuário
Por favor, rotule as tags de tarefa para a query do usuário. Você precisará analisar a query do usuário e selecionar a tag de tarefa mais relevante da lista abaixo.

todas_as_tags_de_tarefa = [
    "Busca de informações", # Usuários pedem informações ou fatos específicos sobre vários tópicos.
    "Raciocínio", # Queries que exigem pensamento lógico, resolução de problemas ou processamento de ideias complexas.
    "Planejamento", # Usuários precisam de assistência na criação de planos ou estratégias para atividades e projetos.
    "Edição", # Envolve edição, reformulação, revisão ou outras tarefas relacionadas à composição de conteúdo escrito geral.
    "Codificação e Debugging", # Usuários buscam ajuda para escrever, revisar ou corrigir código em programação.
    "Matemática", # Queries relacionadas a conceitos, problemas e cálculos matemáticos.
    "Role playing", # Usuários se envolvem em cenários que exigem que o ChatGPT adote um personagem ou persona.
    "Análise de dados", # Solicitações envolvem a interpretação de dados, estatísticas ou a execução de tarefas analíticas.
    "Escrita criativa", # Usuários buscam assistência na elaboração de histórias, poemas ou outros textos criativos.
    "Busca de conselhos", # Usuários pedem recomendações ou orientações sobre várias questões pessoais ou profissionais.
    "Brainstorming", # Envolve a geração de ideias, pensamento criativo ou exploração de possibilidades.
    "Outros" # Quaisquer query que não se encaixem nas categorias acima ou sejam de natureza diversa.
]

## Formato de Saída:
Note que você só pode selecionar uma única tag primária. Outras tags aplicáveis podem ser adicionadas à lista de outras tags.
Agora, por favor, gere suas tags abaixo em um formato json, preenchendo os espaços reservados em <...>:
```
{{
    "tag_primaria": "<tag primária>",
    "outras_tags": ["<tag 1>", "<tag 2>", ... ]
}}
```
""".lstrip()


OUTPUT_CLASSIFICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "tag_primaria": {
            "type": "string",
            "enum": [
                "Busca de informações",
                "Raciocínio",
                "Planejamento",
                "Edição",
                "Codificação e Debugging",
                "Matemática",
                "Role playing",
                "Análise de dados",
                "Escrita criativa",
                "Busca de conselhos",
                "Brainstorming",
                "Outros",
            ],
        },
        "outras_tags": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "Busca de informações",
                    "Raciocínio",
                    "Planejamento",
                    "Edição",
                    "Codificação e Debugging",
                    "Matemática",
                    "Role playing",
                    "Análise de dados",
                    "Escrita criativa",
                    "Busca de conselhos",
                    "Brainstorming",
                    "Outros",
                ],
            },
        },
    },
    "required": ["tag_primaria", "outras_tags"],
}


@step(inputs=["conversation"], outputs=["instruction"])
def GetInstruction(inputs: StepInput) -> "StepOutput":
    for input in inputs:
        input["instruction"] = input["conversation"][0]["content"]
    yield inputs


class AssignTags(Task):
    mission: Literal["dificuldade", "qualidade", "classificacao"]

    @property
    def inputs(self) -> List[str]:
        return ["instruction"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        instruction = input["instruction"]

        if self.mission == "dificuldade":
            input_message = INPUT_DIFFICULTY_RATING_TEMPLATE.format(input=instruction)
        elif self.mission == "qualidade":
            input_message = INPUT_QUALITY_RATING_TEMPLATE.format(input=instruction)
        else:
            input_message = INPUT_CLASSIFICATION_TEMPLATE.format(input=instruction)

        return [{"role": "user", "content": input_message}]

    @property
    def outputs(self) -> List[str]:
        if self.mission == "dificuldade":
            return ["intencao", "conhecimento", "dificuldade", "model_name"]

        if self.mission == "qualidade":
            return ["explicacao", "qualidade", "model_name"]

        return ["tag_primaria", "outras_tags", "model_name"]

    def _impute_output(self) -> Dict[str, None]:
        if self.mission == "dificuldade":
            return {"intencao": None, "conhecimento": None, "dificuldade": None}

        if self.mission == "qualidade":
            return {"explicacao": None, "qualidade": None}

        return {"tag_primaria": None, "outras_tags": None}

    def format_output(
        self,
        output: Union[str, None],
        input: Union[Dict[str, Any], None] = None,
    ) -> Dict[str, Any]:
        if output is None:
            return self._impute_output()

        return json.loads(output)


# https://github.com/magpie-align/magpie/blob/b08a80193c92ea7ec329dd9c23d6c23450c283b5/exp/gen_ins.py#L134
def de_md_logits_processor_for_llama3_1(token_ids, logits):
    # Only process the initial logits
    if len(token_ids) == 0:
        logits[2] = -9999.999  # "#": 2,
        logits[567] = -9999.999  # "##": 567,
        logits[14711] = -9999.999  # "###": 14711,
        logits[827] = -9999.999  # "####": 827,
        logits[334] = -9999.999  # "**": 334
        logits[3146] = -9999.999  # " **": 3146
        logits[96618] = -9999.99  # "**:": 96618

    return logits

def de_md_logits_processor_for_qwen3(token_ids, logits):
    # Only process the initial logits
    if len(token_ids) == 0:
        logits[2] = -9999.999  # "#": 2,
        logits[565] = -9999.999  # "##": 567,
        logits[14374] = -9999.999  # "###": 14711,
        logits[820] = -9999.999  # "####": 827,
        logits[334] = -9999.999  # "**": 334
        logits[3070] = -9999.999  # " **": 3146
        logits[95518] = -9999.99  # "**:": 96618

    return logits


with Pipeline(name="magpie-ultra-pt-v1.0") as pipeline:
    generate_instructions = MagpieGenerator(
        llm=vLLM(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
            tokenizer="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
            magpie_pre_query_template="qwen3",
            extra_kwargs={
                "tensor_parallel_size": 8,
                "max_model_len": 8192,
                "enable_prefix_caching": True,
            },
            generation_kwargs={
                "temperature": 0.7,
                "top_p": 0.8,
                "max_new_tokens": 1024,
                "stop": [
                    "<|im_end|>",      # Main conversation end token
                    "<|endoftext|>",   # End of text token
                    "<|im_start|>",    # Prevents generating new conversation turns
                ],
                "stop_token_ids": [
                    151645,  # <|im_end|>
                    151643,  # <|endoftext|> 
                    151644,  # <|im_start|>
                ],
                "logits_processors": [de_md_logits_processor_for_llama3_1],
            },
        ),
        system_prompt=CATEGORIES_SYSTEM_PROMPTS,
        batch_size=250,
        n_turns=3,
    )

    get_instruction = GetInstruction(input_batch_size=5000)

    assign_difficulty = AssignTags(
        mission="dificuldade",
        llm=vLLM(
            model="Qwen/Qwen3-8B",
            extra_kwargs={
                "tensor_parallel_size": 1,
            },
            structured_output={
                "format": "json",
                "schema": OUTPUT_DIFFICULTY_JSON_SCHEMA,
            },
        ),
        output_mappings={"model_name": "model_name_difficulty"},
        input_batch_size=1000,
    )

    assign_quality = AssignTags(
        mission="qualidade",
        llm=vLLM(
            model="Qwen/Qwen3-8B",
            extra_kwargs={
                "tensor_parallel_size": 1,
            },
            structured_output={
                "format": "json",
                "schema": OUTPUT_QUALITY_JSON_SCHEMA,
            },
        ),
        output_mappings={"model_name": "model_name_quality"},
        input_batch_size=1000,
    )

    assign_classification = AssignTags(
        mission="classificacao",
        llm=vLLM(
            model="Qwen/Qwen3-8B",
            extra_kwargs={
                "tensor_parallel_size": 1,
            },
            structured_output={
                "format": "json",
                "schema": OUTPUT_CLASSIFICATION_JSON_SCHEMA,
            },
        ),
        output_mappings={"model_name": "model_name_classification"},
        input_batch_size=1000,
    )

    embeddings = EmbeddingGeneration(
        embeddings=SentenceTransformerEmbeddings(
            model="Alibaba-NLP/gte-large-en-v1.5",
            device="cuda",
            trust_remote_code=True,
        ),
        input_mappings={"text": "instruction"},
        output_mappings={"model_name": "model_name_embeddings"},
        input_batch_size=50,
    )

    reward_model_score = RewardModelScore(
        model="RLHFlow/ArmoRM-Llama3-8B-v0.1",
        device_map="auto",
        trust_remote_code=True,
        input_batch_size=20,
    )

    combine_outputs = CombineOutputs()

    guard = ChatGeneration(
        llm=vLLM(
            model="meta-llama/Llama-Guard-3-8B",
            extra_kwargs={
                "tensor_parallel_size": 1,
            },
            structured_output={
                "format": "regex",
                "schema": r"\n\n(?:safe|unsafe\n(?:S(?:[1-9]|1[0-4])))",
            },
        ),
        input_mappings={"messages": "conversation"},
        output_mappings={"generation": "guard", "model_name": "model_name_guard"},
        input_batch_size=1000,
    )

    nearest_neighbours = FaissNearestNeighbour(
        metric_type=faiss.METRIC_INNER_PRODUCT, k=5
    )

    (
        generate_instructions
        >> get_instruction
        >> [
            assign_difficulty,
            assign_quality,
            assign_classification,
            embeddings,
            reward_model_score,
            guard,
        ]
        >> combine_outputs
        >> nearest_neighbours
    )


if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            generate_instructions.name: {"num_rows": 1000000, "resources": {"gpus": 8}},
            assign_difficulty.name: {
                "llm": {
                    "generation_kwargs": {"max_new_tokens": 512, "temperature": 0.0}
                },
                "resources": {"gpus": 1},
            },
            assign_quality.name: {
                "llm": {
                    "generation_kwargs": {"max_new_tokens": 512, "temperature": 0.0}
                },
                "resources": {"gpus": 1},
            },
            assign_classification.name: {
                "llm": {
                    "generation_kwargs": {"max_new_tokens": 512, "temperature": 0.0}
                },
                "resources": {"gpus": 1},
            },
            embeddings.name: {
                "resources": {"gpus": 1},
            },
            reward_model_score.name: {"resources": {"gpus": 1, "replicas": 3}},
            guard.name: {
                "llm": {
                    "generation_kwargs": {"max_new_tokens": 128, "temperature": 0.0}
                },
                "resources": {"gpus": 1},
            },
        },
    )

    distiset.push_to_hub("argilla/magpie-ultra-pt-v1.0")
