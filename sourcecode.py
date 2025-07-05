import json
import tempfile
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import traceback  # Для вывода стека ошибок
import re

app = FastAPI(
    title="LangChain Dynamic Generator",
    description="Generate and run LangChain-based applications dynamically.",
    version="1.0"
)

# Middleware для CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_partial_variables(prompt_config):
    """
    Извлекает все partial_variables из prompt_config.json.
    :param prompt_config: Данные из JSON-файла промпта.
    :return: Список названий partial_variables.
    """
    partial_variables = set()

    # Ищем паттерны вида {variable_name} во всех строках
    pattern = r"\{(\w+)}"

    for key, value in prompt_config.items():
        if isinstance(value, str):
            matches = re.findall(pattern, value)
            partial_variables.update(matches)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    matches = re.findall(pattern, item)
                    partial_variables.update(matches)

    return partial_variables

def resolve_mapping(variable, mapping_config, order_config):
    """
    Разрешает значение для переменной на основе mapping_config и order_config.
    :param variable: Название переменной.
    :param mapping_config: Связующий JSON между prompt_config и order_config.
    :param order_config: Данные из JSON-файла order_config.
    :return: Значение переменной.
    """
    mapping_path = mapping_config.get(variable)
    if not mapping_path:
        return f"Value for {variable} not found"

    keys = mapping_path.split(".")
    value = order_config
    try:
        for key in keys:
            if isinstance(value, list):
                value = [item.get(key) for item in value]
            else:
                value = value.get(key)
        return value
    except (AttributeError, KeyError, TypeError):
        return f"Value for {variable} not found"

def map_partial_variables_to_config(partial_variables, mapping_config, order_config):
    """
    Связывает partial_variables из prompt_config с данными из order_config через mapping_config.
    :param partial_variables: Список partial_variables из prompt_config.
    :param mapping_config: Связующий JSON между prompt_config и order_config.
    :param order_config: Данные из JSON-файла order_config.
    :return: Словарь с ключами partial_variables и значениями из order_config.
    """
    mapping = {}
    for variable in partial_variables:
        resolved_value = resolve_mapping(variable, mapping_config, order_config)
        if variable == "details_list":
            # Форматируем details_list
            mapping[variable] = format_details_list(resolved_value)
        elif isinstance(resolved_value, list):
            mapping[variable] = "\n".join(map(str, resolved_value))
        else:
            mapping[variable] = resolved_value
    return mapping

def format_details_list(parameters):
    """
    Форматирует список вопросов для отображения в details_list.
    :param parameters: Список вопросов из survey_config['survey']['questions'].
    :return: Строка с форматированными деталями.
    """
    formatted_details = []
    for param in parameters:
        question_text = param["question"]
        if "options" in param:
            options = ", ".join([f"'{opt}'" for opt in param["options"]])
            formatted_details.append(f"  - {question_text}. Options: {options}.")
        else:
            formatted_details.append(f"  - {question_text}.")
    return "\n".join(formatted_details)


def update_code_with_mappings(code, mapping_config):
    """
    Заменяет все статические ссылки в коде на динамические переменные из mapping_config.
    :param code: Исходный код приложения.
    :param mapping_config: Связующий JSON между prompt_config и order_config.
    :return: Обновленный код.
    """
    for variable, path in mapping_config.items():
        code = code.replace(variable, path)
    return code

def generate_prompt_template(prompt_config, partial_mapping):
    """
    Генерирует корректный шаблон промпта на основе prompt_config и partial_mapping.
    :param prompt_config: Данные из JSON-файла промпта.
    :param partial_mapping: Словарь сопоставления partial_variables.
    :return: Сформированный шаблон промпта.
    """
    prompt = []

    # Добавляем блок character
    prompt.append(f"# character: {prompt_config['character']}\n")

    # Добавляем блок skills
    prompt.append("# skills:\n")
    for skill in prompt_config["skills"]:
        formatted_skill = skill.format(**partial_mapping)
        prompt.append(f"    - {formatted_skill}\n")

    # Добавляем блок constraints
    prompt.append("\n# constraints:\n")
    for constraint in prompt_config["constraints"]:
        prompt.append(f"    - {constraint}\n")

    # Добавляем блок presentation, controller, role
    prompt.append(f"\n# presentation: {prompt_config['presentation']}\n")
    prompt.append(f"# controller: {prompt_config['controller']}\n")
    prompt.append(f"# role: {prompt_config['role'].format(**partial_mapping)}\n")
    
    prompt.append(f"\n# user_interaction:\nPrevious conversation: {{history}}")
    prompt.append(f"\n# response_logic:\nBased on the above interaction and current order details, determine the missing fields.\nIf all fields are complete, confirm the order and request user confirmation.\nIf fields are incomplete, ask targeted questions to gather the remaining details.\nUser input: {{user_input}}")

    return "".join(prompt)

def generate_process_logic(template_type, service_call):
    """
    Генерирует код для маршрута /process в зависимости от типа шаблона и добавляет логику вызова сервиса.
    :param template_type: Тип системы (e.g., "order", "survey").
    :param service_call: Сгенерированный код для вызова сервиса.
    :return: Строка с соответствующей логикой.
    """
    if template_type == "order":
        return f"""
        standardized_data = current_instance.to_dict()
        if current_instance.is_order_complete():
            current_instance.is_complete = True
            current_instance.save_to_json()
            response = "Order complete and saved."
            try:
                {service_call}
            except Exception as e:
                service_response = {{"error": str(e)}}
        else:
            response = chat_chain.invoke({{
                "history": memory.load_memory_variables({{}}).get("history", ""),
                "user_input": user_input
            }})["text"]
            service_response = None
        """
    elif template_type == "survey":
        return f"""
        current_question = current_instance.get_current_question()
        if current_question is None:
            response = "Survey complete."
            current_instance.save_to_json()
            try:
                {service_call}
            except Exception as e:
                service_response = {{"error": str(e)}}
        else:
            response = f"Next question: {{current_question}}"
            current_instance.record_response(current_instance.current_question, user_input)
            current_instance.next_question()
            service_response = None
        """
    else:
        raise ValueError(f"Unsupported template type: {template_type}")


def generate_service_call(service_config, partial_mapping):
    """
    Генерирует код для обращения к сервису на основе service_config.
    :param service_config: Конфигурация сервиса из JSON.
    :param partial_mapping: Сопоставленные значения partial_variables.
    :return: Сгенерированный код обращения к сервису.
    """
    payload = json.dumps(partial_mapping).replace('"', '\\"')

    service_call = f"""
    response = requests.{service_config['service']['method'].lower()}(
        url="{service_config['service']['endpoint']}",
        json={payload}
    )
    if response.status_code == 200:
        return response.json()
    else:
        return {{"error": "Failed to connect to service", "status_code": response.status_code}}
    """
    return service_call


def generate_langchain_code(prompt_json, config_json, mapping_json, service_json, template_type="order"):
    """
    Генерирует код для системы на LangChain на основе входных JSON файлов и типа шаблона.
    :param prompt_json: Путь к JSON-файлу с промптом.
    :param config_json: Путь к JSON-файлу с параметрами конфигурации.
    :param mapping_json: Путь к JSON-файлу с маппингом.
    :param template_type: Тип системы (e.g., "order", "survey").
    :return: Сгенерированный Python-код в виде строки.
    """
    # Загрузка JSON файлов
    with open(prompt_json, "r") as file:
        prompt_config = json.load(file)

    with open(config_json, "r") as file:
        order_config = json.load(file)

    with open(mapping_json, "r") as file:
        mapping_config = json.load(file)
    
    with open(service_json, "r") as file:
        service_config = json.load(file)

    # Извлечение partial_variables и связывание с данными
    partial_variables = extract_partial_variables(prompt_config)
    partial_mapping = map_partial_variables_to_config(partial_variables, mapping_config, order_config)

    # Генерация шаблона промпта
    base_prompt_template = generate_prompt_template(prompt_config, partial_mapping)
    
    service_call_code = generate_service_call(service_config, partial_mapping)
    process_logic_code = generate_process_logic(template_type, service_call_code)

    # Генерация класса на основе типа шаблона
    if template_type == "order":
        dynamic_class = """
class DynamicOrder:
    def __init__(self, config):
        self.fields = {}
        self.required_fields = []
        self.collection_fields = []
        self.is_complete = False

        for param in config["product"]["parameters"]:
            field_name = param["name"].lower()
            self.fields[field_name] = [] if param["is_collection"] == "True" else None
            if param["is_required"] == "True":
                self.required_fields.append(field_name)
            if param["is_collection"] == "True":
                self.collection_fields.append(field_name)

    def update_order(self, key, value):
        if key in self.fields:
            if key in self.collection_fields:
                if isinstance(self.fields[key], list):
                    if value not in self.fields[key]:
                        self.fields[key].append(value)
            else:
                self.fields[key] = value

    def is_order_complete(self):
        return all(self.fields[field] for field in self.required_fields)

    def to_dict(self):
        return {key: value for key, value in self.fields.items()} | {"is_complete": self.is_complete}

    def save_to_json(self, filename="output.json"):
        with open(filename, "w") as file:
            json.dump(self.to_dict(), file, indent=4)
"""
    elif template_type == "survey":
        dynamic_class = """
class DynamicSurvey:
    def __init__(self, config):
        self.questions = config["survey"]["questions"]
        self.responses = {}
        self.current_question = 0

    def record_response(self, question_id, response):
        self.responses[question_id] = response

    def next_question(self):
        if self.current_question < len(self.questions):
            self.current_question += 1
        return self.get_current_question()

    def get_current_question(self):
        if self.current_question < len(self.questions):
            return self.questions[self.current_question]
        return None

    def is_survey_complete(self):
        return self.current_question >= len(self.questions)

    def to_dict(self):
        return {"responses": self.responses, "is_complete": self.is_survey_complete()}
"""
    else:
        raise ValueError(f"Unsupported template type: {template_type}")

    class_type = {
        "order": "DynamicOrder(config)",
        "survey": "DynamicSurvey(config)"
    }

    # Генерация основного кода
    main_code = f"""
from langchain_community.llms.ollama import Ollama
from fastapi import FastAPI
from langserve import add_routes
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from pydantic import BaseModel
import json
import requests

# Загрузка конфигурации из JSON
config = {json.dumps(order_config)}

# Создание экземпляра
current_instance = {class_type[template_type]}

# Инициализация FastAPI
app = FastAPI(
    title="{prompt_config['controller']}",
    version="1.0",
    description="{prompt_config['presentation']}",
)

# Подключение к модели
llm = Ollama(model="llama3", temperature=0.1)

# Создание памяти для хранения промежуточных ответов
memory = ConversationBufferMemory()

# Шаблон промпта
base_prompt_template = \"\"\"{base_prompt_template}\"\"\"

prompt_template = PromptTemplate(
    template=base_prompt_template,
    input_variables=["history", "user_input"],
)

# Инициализация LangChain
chat_chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

add_routes(app, chat_chain, path="/process")

class UserMessage(BaseModel):
    message: str

@app.post("/process")
async def process_request(user_message: UserMessage):
    user_input = user_message.message
    memory.chat_memory.add_user_message(user_input)
    service_response = None

    {process_logic_code}
    
    memory.chat_memory.add_ai_message(response)
    return {{
        "response": response,
        "data": current_instance.to_dict(),
        "service_response": service_response
    }}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080)
"""

    # Объединение всего кода
    full_code = dynamic_class + main_code
    
    final_code = update_code_with_mappings(full_code, mapping_config)
    
    return final_code


@app.post("/generate")
async def generate_code(
    prompt_file: UploadFile,
    config_file: UploadFile,
    mapping_file: UploadFile,
    service_file: UploadFile,
    template_type: str = Form(...)
):
    try:
        # Используем временные файлы через tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as prompt_temp:
            prompt_path = prompt_temp.name
            prompt_temp.write(await prompt_file.read())
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as config_temp:
            config_path = config_temp.name
            config_temp.write(await config_file.read())
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as mapping_temp:
            mapping_path = mapping_temp.name
            mapping_temp.write(await mapping_file.read())
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as service_temp:
            service_path = service_temp.name
            service_temp.write(await service_file.read())

        # Импортируем функцию генерации
        generated_code = generate_langchain_code(
            prompt_json=prompt_path,
            config_json=config_path,
            mapping_json=mapping_path,
            service_json=service_path,
            template_type=template_type
        )

        # Удаляем временные файлы
        os.unlink(prompt_path)
        os.unlink(config_path)
        os.unlink(mapping_path)
        os.unlink(service_path)

        # Возвращаем сгенерированный код
        return JSONResponse(content={"status": "success", "code": generated_code})
    except Exception as e:
        # Логируем ошибки
        error_message = traceback.format_exc()
        print(f"Ошибка на сервере: {error_message}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
