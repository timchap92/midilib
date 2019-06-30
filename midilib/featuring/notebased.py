import json
from logging import warning

import numpy as np

import midilib
from midilib import util
from midilib.featuring.util import constrain_pitch
from midilib.util import registered_tqdm



class FeaturedNote:

    def __init__(self, pitch, wait, nb_pitches):
        self.pitch = pitch
        self.wait = wait
        self.nb_pitches = nb_pitches

    @classmethod
    def create_with_pitch_clipping(cls, pitch, wait, min_pitch, max_pitch):
        nb_pitches = max_pitch - min_pitch + 1
        if pitch is None:
            # Starting note, use special out-of-bounds value
            pitch = max_pitch - min_pitch
        else:
            pitch = constrain_pitch(pitch, min_pitch, max_pitch) - min_pitch
            assert 0 <= pitch < max_pitch - min_pitch

        return cls(pitch, wait, nb_pitches)

    def __repr__(self):
        return 'FeaturedNote(pitch={pitch}, wait={wait})'.format(**self.__dict__)

    def to_tuple(self):
        return self.pitch, self.wait, self.nb_pitches

    @classmethod
    def from_tuple(cls, tpl):
        return cls(*tpl)

    def calculate_features(self):
        pitch_features = [0] * self.nb_pitches
        pitch_features[self.pitch] = 1
        self.features = np.array(pitch_features + [self.wait])
        self.pitch_label = np.array(pitch_features)


class FeaturingConfig:

    def __init__(self, min_pitch, max_pitch, nb_notes_history):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.nb_notes_history = nb_notes_history
        self.nb_pitches = max_pitch - min_pitch + 1  # including out of bounds pitch
        self.nb_features = self.nb_pitches + 1  # including scalar continuous wait feature

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, dct):
        return cls(dct['min_pitch'], dct['max_pitch'], dct['nb_notes_history'])


class NoteBasedFeaturer:

    def __init__(self, featuring_config):
        self.featuring_config = featuring_config

    def feature_notes(self, song):

        first_note = FeaturedNote.create_with_pitch_clipping(
            pitch=None, wait=0,
            min_pitch=self.featuring_config.min_pitch,
            max_pitch=self.featuring_config.max_pitch)

        featured_notes = [first_note for _ in range(self.featuring_config.nb_notes_history)]
        previous_time = -1

        for note in song:
            wait = note.start - previous_time

            fnote = FeaturedNote.create_with_pitch_clipping(
                note.pitch,
                wait,
                min_pitch=self.featuring_config.min_pitch,
                max_pitch=self.featuring_config.max_pitch)

            featured_notes.append(fnote)
            previous_time = note.start

        return featured_notes

    def extract_sequences_for_pitch_label(self, featured_songs):
        nb_datapoints = sum(len(fsong) - self.featuring_config.nb_notes_history
                            for fsong in featured_songs)

        sequences = -1 * np.ones(shape=(nb_datapoints, self.featuring_config.nb_notes_history,
                                        self.featuring_config.nb_features))
        pitch_labels = -1 * np.ones(shape=(nb_datapoints, self.featuring_config.nb_pitches))

        data_index = 0

        for fsong in registered_tqdm(featured_songs):

            for fnote in fsong:
                fnote.calculate_features()

            for i in range(self.featuring_config.nb_notes_history, len(fsong)):
                sequences[data_index] = np.array(
                    [fnote.features
                     for fnote in fsong[i - self.featuring_config.nb_notes_history:i]]
                )
                pitch_labels[data_index] = fsong[i].pitch_label
                data_index += 1

        return sequences, pitch_labels

    def extract_sequences_for_wait_label(self, featured_songs):

        nb_datapoints = sum(len(fsong) - self.featuring_config.nb_notes_history
                            for fsong in featured_songs)

        sequences = -1 * np.ones(shape=(nb_datapoints, self.featuring_config.nb_notes_history,
                                        self.featuring_config.nb_features))
        pitches = -1 * np.ones(shape=(nb_datapoints, self.featuring_config.nb_pitches))
        waits = -1 * np.ones(shape=(nb_datapoints,))

        data_index = 0

        for fsong in registered_tqdm(featured_songs):

            for fnote in fsong:
                fnote.calculate_features()

            for i in range(self.featuring_config.nb_notes_history, len(fsong)):
                sequences[data_index] = np.array(
                    [fnote.features
                     for fnote in fsong[i - self.featuring_config.nb_notes_history:i]]
                )
                pitches[data_index] = fsong[i].pitch_label
                waits[data_index] = fsong[i].wait
                data_index += 1

        return sequences, pitches, waits

    def extract_sequence_for_prediction(self, featured_notes):
        sequences = -1 * np.ones(shape=(1, self.featuring_config.nb_notes_history,
                                        self.featuring_config.nb_features))

        sequences[0] = np.array(
            [fnote.features
             for fnote in featured_notes[-self.featuring_config.nb_notes_history:]]
        )

        return sequences

    def to_path(self, path):
        try:
            code_version = midilib.__version__
        except AttributeError:
            code_version = 'unknown'

        dct = {
            'midilib_version': code_version,
            'type': self.__class__.__name__,
            'featuring_config': self.featuring_config.to_dict()
        }

        if path.startswith('gs://'):
            util.dump_string_to_gcs(json.dumps(dct))
        with open(path) as f:
            json.dump(dct, f)

    @classmethod
    def from_path(cls, path):
        try:
            code_version = midilib.__version__
        except AttributeError:
            code_version = 'unknown'

        with open(path) as f:
            dct = json.load(f)

        if dct['version'] != code_version:
            warning('Code versions do not match. Saved: {} Current: {}'.format(
                dct['version'],
                code_version))

        assert dct['type'] == __class__.__name__

        return cls(FeaturingConfig.from_dict(dct['featuring_config']))
