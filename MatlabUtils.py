import matlab.engine


engine = None


def find_engine():
    global engine

    if engine is not None:
        return engine

    engines = matlab.engine.find_matlab()
    print(engines)

    if engines:
        print("Found MATLAB engine! connecting ...")
        engine = matlab.engine.connect_matlab(engines[0])
        print("Found MATLAB engine! connected")
    else:
        print("No MATLAB engine found, launching new one ...")
        engine = matlab.engine.start_matlab()
        print("No MATLAB engine found, launched new one")

    engine.addpath('amsbss/ILRMA')
    engine.addpath('amsbss/AuxIVA')
    engine.addpath('amsbss/AuxIVA/STFT')
    return engine


def stop_engine(engine):
    if engine is not None:
        engine.quit()
