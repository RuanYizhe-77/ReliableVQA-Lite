from src.confidence.verifier import parse_verifier_output


def test_verifier_parser_handles_unquoted_confidence_word():
    raw = '''
```json
{
  "supported": true,
  "confidence": high,
  "explanation": "The image supports it."
}
```
'''
    parsed = parse_verifier_output(raw)
    assert parsed.supported is True
    assert parsed.confidence == 0.8
    assert parsed.parse_ok is False


def test_verifier_parser_prefers_unsupported_in_fallback():
    parsed = parse_verifier_output('{"supported": false, "confidence": medium}')
    assert parsed.supported is False
    assert parsed.confidence == 0.5
