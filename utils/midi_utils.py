class MidiUtils:
    def __init__(self):
        pass

    def _program_to_group(self, program):  # midi program number to instrument group specified in oneshot dataset
        if program <= 7:
            return "keyboard"
        if 8 <= program <= 15:
            return "mallet"
        if 16 <= program <= 23:
            return "organ"
        if 24 <= program <= 31 or 104 <= program <= 111:
            return "guitar"
        if 32 <= program <= 39:
            return "bass"
        if 40 <= program <= 52 or program == 56:
            return "string"
        if 56 <= program <= 63:
            return "brass"
        if 64 <= program <= 71:
            return "reed"
        if 72 <= program <= 79:
            return "flute"
        if 80 <= program <= 104:
            return "synth"
        if program in [53, 54, 55, 86]:
            return "vocal"
        else:
            return "other"

    def valid_note_per_instrument(self, instrument, pitch):
        def no_valid_note(instrument):
            raise NotImplementedError(f"{instrument} is not supported yet")

        if instrument == "drums":
            if pitch < 35 or pitch > 81:
                return False

        # TODO add all the instruments
        elif instrument == "keyboard":
            no_valid_note(instrument)
        elif instrument == "mallet":
            no_valid_note(instrument)
        elif instrument == "organ":
            no_valid_note(instrument)
        elif instrument == "guitar":
            no_valid_note(instrument)
        elif instrument == "bass":
            no_valid_note(instrument)
        elif instrument == "string":
            no_valid_note(instrument)
        elif instrument == "brass":
            no_valid_note(instrument)
        elif instrument == "reed":
            no_valid_note(instrument)
        elif instrument == "flute":
            no_valid_note(instrument)
        elif instrument == "synth":
            no_valid_note(instrument)
        elif instrument == "vocal":
            no_valid_note(instrument)
        else:
            return False
        return True

    def invalid_drum_note(self, pitch: int, onset: float, offset: float):
        return pitch > 81 or pitch < 35 or onset >= offset

    def invalid_instrument_note(self, pitch: int, onset: float, offset: float):
        return pitch > 127 or pitch < 0 or onset >= offset

    def offset_length_check(self, onset: float, offset: float):
        if offset - onset < 0.01:
            offset += 0.01  # For extremely short notes (e.g., drums), prevent onset and offset token times from being identical (duration=0)
        return offset
