class MappingUtils:
    def __init__(self):
        self.GM_standard_midi_to_Gm_custom_Mapping = {
            35: 35,  # Acoustic Bass Drum -> # Acoustic Bass Drum
            36: 36,  # Bass Drum 1 -> # Bass Drum 1
            37: 37,  # Side Stick -> # Side Stick
            38: 38,  # Acoustic Snare -> # Acoustic Snare
            39: 39,  # Hand Clap -> # Hand Clap
            40: 40,  # Electric Snare -> # Electric Snare
            41: 41,  # Floor Tom -> # Floor Tom
            42: 42,  # Closed Hi Hat -> # Closed Hi Hat
            43: 41,  # High floor tom -> # Floor Tom
            44: 43,  # Pedal Hi-Hat -> # Pedal Hi-Hat
            45: 41,  # Low Tom -> # Floor Tom
            46: 44,  # Open Hi-Hat -> # Open Hi-Hat
            47: 45,  # Low Mid Tom -> # Mid Tom
            48: 45,  # High Mid Tom -> # Mid Tom
            49: 46,  # Crash Cymbal -> # Crash Cymbal
            50: 47,  # High Tom -> # High Tom
            51: 48,  # Ride Cymbal -> # Ride Cymbal
            52: 49,  # Chinese Cymbal -> # Chinese Cymbal
            53: 48,  # Ride Bell -> # Ride Cymbal
            54: 50,  # Tambourine -> # Tambourine
            55: 51,  # Splash Cymbal -> # Splash Cymbal
            56: 52,  # Cowbell -> # Cowbell
            57: 46,  # Crash Cymbal 2 -> # Crash Cymbal
            58: 53,  # Vibraslap -> # Vibraslap
            59: 48,  # Ride Cymbal 2 -> # Ride Cymbal
            60: 54,  # Hi Bongo -> # Congas & Timbales
            61: 54,  # Low Bongo -> # Congas & Timbales
            62: 54,  # Mute Hi Conga -> # Congas & Timbales
            63: 54,  # Open Hi Conga -> # Congas & Timbales
            64: 54,  # Low Conga -> # Congas & Timbales
            65: 54,  # High Timbale -> # Congas & Timbales
            66: 54,  # Low Timbale -> # Congas & Timbales
            67: 52,  # High Agogo -> # Cowbell
            68: 52,  # Low Agogo -> # Cowbell
            69: 55,  # Cabasa -> # Shaker
            70: 55,  # Maracas -> # Shaker
            71: 56,  # Short Whistle -> # Whistle
            72: 56,  # Long Whistle -> # Whistle
            73: 57,  # Short Guiro -> # Guiro
            74: 57,  # Long Guiro -> # Guiro
            75: 58,  # Claves -> # Claves
            76: 58,  # Hi Wood Block -> # Claves
            77: 58,  # Low Wood Block -> # Claves
            78: 59,  # Mute Cuica -> # Cuica
            79: 59,  # Open Cuica -> # Cuica
            80: 60,  # Mute Triangle -> # Triangle
            81: 60,  # Open Triangle -> # Triangle
        }
        self.GM_custom_midi_to_Gm_standard_Mapping = {
            v: k for k, v in self.GM_standard_midi_to_Gm_custom_Mapping.items()
        }

        self.ADTOF_mapping = {
            35: 35,  # Acoustic Bass Drum -> # Acoustic Bass Drum
            36: 35,  # Bass Drum 1 -> # Acoustic Bass Drum
            37: 38,  # Side Stick -> # Acoustic Snare
            38: 38,  # Acoustic Snare -> # Acoustic Snare
            39: 38,  # Hand Clap -> # Acoustic Snare
            40: 38,  # Electric Snare -> # Acoustic Snare
            41: 41,  # Floor Tom -> # Floor Tom
            42: 42,  # Closed Hi Hat -> # Closed Hi Hat
            43: 42,  # Pedal Hi-Hat -> # Closed Hi Hat
            44: 42,  # Open Hi-Hat -> # Closed Hi Hat
            45: 41,  # Mid Tom -> # Floor Tom
            46: 48,  # Crash Cymbal -> # Ride Cymbal
            47: 41,  # High Tom -> # Floor Tom
            48: 48,  # Ride Cymbal -> # Ride Cymbal
            49: 48,  # Chinese Cymbal -> # Ride Cymbal
            50: 42,  # Tambourine -> # Closed Hi Hat
            51: 48,  # Splash Cymbal -> # Ride Cymbal
            52: 52,  # Cowbell -> # Cowbell
            53: 61,  # Vibraslap -> # Other
            54: 61,  # Congas & Timbales -> # Other
            55: 61,  # Shaker -> # Other
            56: 61,  # Whistle -> # Other
            57: 61,  # Guiro -> # Other
            58: 58,  # Claves -> # Claves
            59: 61,  # Cuica -> # Other
            60: 61,  # Triangle -> # Other
            61: 61,  # Other -> # Other
        }

        self.ADTOF_inverse_mapping = {
            35: [35, 36],
            38: [37, 38, 39, 40],
            41: [41, 45, 47],
            42: [42, 43, 44, 50],
            48: [46, 48, 49, 51],
            52: [52],
            58: [58],
            61: [53, 54, 55, 56, 57, 59, 60],
        }

        self.ADTOF_label_mapping = {
            35: "BD",
            38: "SD",
            41: "TT",
            42: "HH",
            48: "CY + RD",
            52: "Cowbell",
            58: "Claves",
            61: "Other",
        }
        self.ADTOF_label_to_midi_mapping = {
            "BD": 35,
            "SD": 38,
            "TT": 41,
            "HH": 42,
            "CY + RD": 48,
            "Cowbell": 52,
            "Claves": 58,
            "Other": 61,
        }
        self.GM_reduced_name_convention = {
            35: "Acoustic Bass Drum",
            36: "Bass Drum 1",
            37: "Side Stick",
            38: "Acoustic Snare",
            39: "Hand Clap",
            40: "Electric Snare",
            41: "Floor Tom",
            42: "Closed Hi Hat",
            43: "Pedal Hi-Hat",
            44: "Open Hi-Hat",
            45: "Mid Tom",
            46: "Crash Cymbal",
            47: "High Tom",
            48: "Ride Cymbal",
            49: "Chinese Cymbal",
            50: "Tambourine",
            51: "Splash Cymbal",
            52: "Cowbell",
            53: "Vibraslap",
            54: "Congas & Timbales",
            55: "Shaker",
            56: "Whistle",
            57: "Guiro",
            58: "Claves",
            59: "Cuica",
            60: "Triangle",
        }
        self.MDB_to_Standard_MIDI = {
            "KD": 35,  # Acoustic Bass Drum
            "SD": 38,  # Acoustic Snare
            "SDB": 38,  # Acoustic Snare
            "SDD": 38,  # Acoustic Snare
            "SDF": 38,  # Acoustic Snare
            "SDG": 38,  # Acoustic Snare
            "SDNS": 38,  # Acoustic Snare
            "CHH": 42,  # Closed Hi Hat
            "OHH": 46,  # Open Hi Hat
            "PHH": 44,  # Pedal Hi Hat
            "HIT": 50,  # High Tom
            "MHT": 48,  # High Mid Tom
            "HFT": 43,  # High Floor Tom
            "LFT": 41,  # Low Floor Tom
            "RDC": 51,  # Ride Cymbal
            "RDB": 53,  # Ride Bell
            "CRC": 49,  # Crash Cymbal
            "CHC": 52,  # Chinese Cymbal
            "SPC": 55,  # Splash Cymbal
            "SST": 37,  # Side Stick
            "TMB": 54,  # Tambourine
        }
        self.ENST_to_Standard_MIDI = {
            "bd": 35,  # Acoustic Bass Drum
            "cs": 37,  # Side Stick
            "sweep": 38,  # Acoustic Snare
            "rs": 38,  # Acoustic Snare
            "sd": 38,  # Acoustic Snare
            "sd-": 38,  # Acoustic Snare
            "lft": 41,  # Low Floor Tom
            "chh": 42,  # Closed Hi-Hat
            "lt": 45,  # Low Tom
            "ltr": 45,  # Low Tom
            "ohh": 46,  # Open Hi-Hat
            "lmt": 47,  # Low Mid Tom
            "mt": 48,  # Hi-Mid Tom
            "mtr": 48,  # Hi-Mid Tom
            "cr": 49,  # Crash Cymbal 1
            "c1": 49,  # Crash Cymbal 1
            "cr1": 49,  # Crash Cymbal 1
            "cr5": 49,  # Crash Cymbal 1
            "rc": 51,  # Ride Cymbal 1
            "rc1": 51,  # Ride Cymbal 1
            "rc3": 51,  # Ride Cymbal 1
            "ch": 52,  # Chinese Cymbal
            "ch1": 52,  # Chinese Cymbal
            "ch5": 52,  # Chinese Cymbal
            "spl": 55,  # Splash Cymbal
            "spl2": 55,  # Splash Cymbal
            "cb": 56,  # Cowbell
            "cr2": 57,  # Crash Cymbal 2
            "c": 57,  # Crash Cymbal 2
            "c4": 57,  # Crash Cymbal 2
            "rc2": 59,  # Ride Cymbal 2
            "rc4": 59,  # Ride Cymbal 2
            "sticks": 75,  # Claves
        }

        self.TMIDT_to_Standard_MIDI = {
            0: 35,  # Acoustic Bass Drum
            1: 38,  # Acoustic Snare
            2: 41,  # Floor Tom
            3: 42,  # Closed Hi Hat
            4: 49,  # Crash Cymbal
            5: 51,  # Ride Cymbal
            6: 53,  # Ride Bell
            7: 75,  # Claves
        }
