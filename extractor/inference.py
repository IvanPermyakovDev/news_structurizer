import argparse
import json
import logging
import os
import sys
import torch
from dataclasses import dataclass
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Пример текста (ASR)
DEFAULT_TEXT = (
    "курметти достар бу-бугин биз тарихи манызы бар жаналыкпен болисемиз ягни туркистан облысында археологиялык казба жумыстары кезинде ежелги каланын орны табылды бул жаңалыкты берик абдигали мен гулнар оспан хабарлады олардын айтуынша табылган жәдігерлер биздин заманымызга деинги кезеңге жатады казыргы танда археологтар сол жерде зерттеу жумыстарын жалгастырып жатыр бул жерден кыштан жасалган буйымдар алтын ашекейлер жане ежелги кару жарактар табылган ме-мемлекеттикбагдарлама аясында мадени мураны коргау жумыстары белсенди жургизилуде бирак кейбир тарихи орындардын жагдайы маз емес деиди мамандар тарихи ескерткиштерди сактап калу ушин косымша каржы жане заманауи технологиялар кажет эээ ягни тарихи сананы калыптастыру ушин осындай олжалардын манызы зор туркистан каласындагы ахмет ясауи кесенесинин манында да зерттеулер жургизилип жатыр баскарма сынын айтуынша бул жер туристер ушин таптырмас орын болмакшы казыргы кезде туризм саласын дамыту басты багыттардын бири болып табылады бирак инфракурылым маселеси али де шешимин таппай отыр жолдардын сапасыздыгы мен меиманханалардын аздыгы кедерги келтируде келе жатырмз осы багытта жумыс истеп жаткан мамандардын бири гулнар оспан онын айтуынша биз оз тарихымызды алемге танытуымыз керек ягни казакстан тек мунай мен газымен гана емес бай тарихымен де мактана алады деиди сарапшылар бү-бүгін биз осы манызды тақырыпты талкылайтын боламыз министерство тарапынан мадениет саласына болинетин каражат жылдан жылга артып келеди бирак онын тиимди жумсалуы басты маселе болып кала береди эээ сонымен катар музейлердин жагдайын жаксарту керек казыргы заманда музейлер тек заттар сакталатын жер емес интерактивти орталык болуы тиис ягни жастарды кызыктыру ушин жана форматтар кажет мектепке барган окушылар тарихты китаптан гана емес осындай орындарды кору аркылы тануы керек балаларге арналган арнайы экскурсиялар уйымдастырылуда бул оте жаксы бастама деиди ата аналар казыргы танда коптеген жәдігерлер шетелдик музейлерде сактаулы тур оларды елге кайтару маселеси де кун тәртібінде тур ягни бул халыкаралык деңгейде шешилетин маселе деуге болады ба-балалар оз тарихын билмесе болашагымыз булынгыр болады сондыктан да тарихты зерттеуге баса назар аударуымыз керек деп-деп айтады берик абдигали онын пикиринше археологиялык зерттеулерге жас галымдарды тарту кажет казыргы кезде галымдардын жалакысы мен алеуметтик жагдайы да назарда болуы тиис эээ ягни гылымсыз даму болмайды бул акикат деуге болады бү-бүгін биз осы салада жумыс истейтін бир топ галыммен кездесу откиздик олардын айтуынша зерттеу жумыстарына кажетти курал жабдыктар жетіспейді ягни шетелдик технологияларды сатып алу ушин улкен каржы керек деиди мамандар сонымен катар тарихи орындарды коршап алу жане оларды вандализмнен коргау маселеси де бар кейбир адамдар тарихи тастарды алып кетеди немесе оларга жазу жазып кетеди бул улкен мадениетсиздик деп есептеймиз ягни халыктын сауаттылыгы мен мадениетин котеру керек казыргы танда арнайы зан жобасы азирленип жатыр онда тарихи жәдігерлерге закым келтиргендер ушин жауапкершилик кушейтилмекши деиди ресми окилдер ягни биз оз кундылыктарымызды коргай алуымыз керек бул биздин парызымыз деуге болады соз сонында айтарымыз тарихты курметтеу отанды суюден басталады сондыктан да осындай жаналыктар кобирек болса иги еди деимиз биз зерттеу жумыстарын бакылап отыратын боламыз келеси шыгарылымдарда жана жәдігерлер туралы акпарат беремиз ягни биздин арнадан алыстаманыздар аман сау болыныздар деи отырып сюжетти аяктагым келеди казыргы кезде мадениет саласындагы реформалар оз натижесин бере бастады деп сенгимиз келеди ягни болашакта биздин тарихымыз алемдик денгейде мойындалатын болады деиди сарапшылар мамандардын айтуынша туркистан каласы рухани астана ретинде дами береди."
)

@dataclass
class GenerationConfig:
    """Параметры генерации текста."""
    max_new_tokens: int = 128
    num_beams: int = 4
    no_repeat_ngram_size: int = 2
    repetition_penalty: float = 1.2
    early_stopping: bool = True

class AttributeExtractor:
    """Класс для извлечения сущностей из текста с использованием обученной модели."""

    TASKS = {
        "title": "тақырып: ",
        "key_events": "оқиға: ",
        "location": "орын: ",
        "key_names": "есімдер: "
    }

    def __init__(self, model_path: str, device: Optional[str] = None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        """Загрузка модели и токенизатора."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        logger.info(f"Loading model from {self.model_path} to {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)

    def extract(self, text: str, config: GenerationConfig) -> Dict[str, str]:
        """
        Извлекает атрибуты из текста.
        """
        results = {}
        logger.info("Starting attribute extraction...")

        for key, prefix in self.TASKS.items():
            input_text = f"{prefix}{text}"
            
            # Токенизация
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            ).to(self.device)

            # Генерация
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=config.max_new_tokens,
                    num_beams=config.num_beams,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                    repetition_penalty=config.repetition_penalty,
                    early_stopping=config.early_stopping
                )

            # Декодинг
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results[key] = decoded_output.strip()

        return results

def main():
    parser = argparse.ArgumentParser(description="Extract attributes from Kazakh news text.")
    parser.add_argument("--model_path", type=str, default="./t5gemma_270m_kz_news_attributes_frozen/checkpoint-1500", help="Path to the trained model")
    parser.add_argument("--text", type=str, help="Input text for extraction")
    parser.add_argument("--file", type=str, help="Path to a text file with input text")
    parser.add_argument("--output", type=str, help="Path to save output JSON")
    
    args = parser.parse_args()

    # Определение входного текста
    input_text = DEFAULT_TEXT
    if args.file:
        if os.path.exists(args.file):
            with open(args.file, 'r', encoding='utf-8') as f:
                input_text = f.read().strip()
            logger.info(f"Loaded text from file: {args.file}")
        else:
            logger.error(f"File not found: {args.file}")
            return
    elif args.text:
        input_text = args.text

    # Конфигурация
    gen_config = GenerationConfig()
    extractor = AttributeExtractor(model_path=args.model_path)

    # Запуск
    logger.info(f"Processing text: {input_text[:100]}...")
    attributes = extractor.extract(input_text, gen_config)

    # Вывод
    json_output = json.dumps(attributes, indent=4, ensure_ascii=False)
    print("\n=== Extraction Result ===")
    print(json_output)
    print("=========================")

    # Сохранение в файл
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(json_output)
        logger.info(f"Result saved to {args.output}")

if __name__ == "__main__":
    main()