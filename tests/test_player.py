from examples.play_model import main


def test_main():
    main(render=True, n_steps=20, n_trials=2, sGL="PIL")
    main(render=True, n_steps=20, n_trials=2, sGL="PILSVG")
