"""Keep OpenRouter unit tests off the network.

``OpenrouterConfig.get_supported_openai_params`` consults OpenRouter's public
model list to decide which reasoning knobs a model accepts. That lookup fails
soft, so tests would still pass without a network, but they would be doing real
HTTP on the way there: slow, dependent on an external service, and different
depending on what OpenRouter published that morning.

Disabling the lookup by default makes every pre-existing test in this directory
exercise the model-cost-map path deterministically. Tests that are specifically
about the live lookup re-enable it and patch the fetch.
"""

import pytest

from litellm.llms.openrouter import capabilities


@pytest.fixture(autouse=True)
def _disable_openrouter_capability_fetch(monkeypatch):
    monkeypatch.setenv("LITELLM_OPENROUTER_CAPABILITY_FETCH", "0")
    capabilities.reset_cache()
    yield
    capabilities.reset_cache()
