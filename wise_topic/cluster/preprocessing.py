def preprocess_nps_data(text: str) -> str:
    # remove the most common non-informative substrings
    text = (
        str(text)
        .replace("Good morning", "")
        .replace("Good afternoon", "")
        .replace("Good evening", "")
        .replace("probable", "likely")
        .replace("Very likely", "")
        .replace("Hello", "")
        .replace("Wise", "")
        .replace("WISE", "")
        .replace("wise", "")  # They're all about Wise anyway
    )
    return text


try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    try:
        import spacy

        spacy.load("en_core_web_lg")
    except:
        ImportError("run python -m spacy download 'en_core_web_lg' first")

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    def anonymize_string(text: str) -> str:
        # https://microsoft.github.io/presidio/supported_entities/
        results = analyzer.analyze(
            text=text,
            entities=[
                "PHONE_NUMBER",
                "CREDIT_CARD",
                "EMAIL_ADDRESS",
                "IBAN_CODE",
                "IP_ADDRESS",
                "PERSON",
            ],
            language="en",
        )
        anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results).text
        out = anonymized_text.split("\n")[0].replace("<PERSON>", "")
        return out

except ImportError:
    # a soft fail: if we don't actually call the function, this will run fine
    def anonymize_string(text: str):
        raise ImportError("Please install presidio_analyzer and presidio_anonymizer first")
