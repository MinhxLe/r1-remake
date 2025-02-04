from r1.data.countdown import compute_score, Task

FMT_SCORE = 0.1
SCORE = 1
TASK: Task = {"nums": [1, 2], "target": 3}


class TestComputeScore:
    def _fmt_response(self, equation_str: str):
        return f"Assistant: <answer>{equation_str}</answer>"

    def compute_score(self, response_str, task={"nums": [1, 2], "target": 3}):
        return compute_score(response_str, task, FMT_SCORE, SCORE)

    def test_invalid_task_response(self):
        assert self.compute_score("invalid response") == 0

    def test_invalid_symbols(self):
        assert self.compute_score(self._fmt_response("1+1!")) == FMT_SCORE

    def test_invalid_numbers(self):
        assert (
            self.compute_score(
                self._fmt_response("1+1"), task=dict(nums=[1, 2], target=3)
            )
            == FMT_SCORE
        )

    def test_wrong_answer(self):
        assert (
            self.compute_score(
                self._fmt_response("1-2"), task=dict(nums=[1, 2], target=3)
            )
            == FMT_SCORE
        )

    def test_right_answer(self):
        assert (
            self.compute_score(
                self._fmt_response("1+2"), task=dict(nums=[1, 2], target=3)
            )
            == SCORE
        )
