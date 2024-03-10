from utilities.parse_arguments import parse_arguments
import pytest

def test_parse_arguments_parses_required():

	test_arguments = ["--model_config", "demollama", "--batch_size", "42"]
	new_arguments = parse_arguments(test_arguments)

	assert new_arguments.model_config == "demollama"
	assert new_arguments.batch_size == 42

def test_parse_arguments_has_default_values():

	test_arguments = ["--model_config", "demollama", "--batch_size", "420"]
	new_arguments = parse_arguments(test_arguments)

	assert new_arguments.seed == 0

def test_parse_arguments_checks_for_torchrun():

	test_arguments = ["--model_config", "demomistrel", "--batch_size", "69"]
	new_arguments = parse_arguments(test_arguments)

	assert "checkpoints/demomistrel" in new_arguments.save_dir

	assert new_arguments.gradient_accumulation == 1
	assert new_arguments.total_batch_size == 69

def test_parse_arguments_total_batch_size_can_be_negative():

	test_arguments = ["--model_config", "demomistrel", "--batch_size", "-1"]
	new_arguments = parse_arguments(test_arguments)
	assert new_arguments.total_batch_size == -1
