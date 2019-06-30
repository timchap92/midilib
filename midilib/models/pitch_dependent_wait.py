import os

import numpy as np
from pretty_midi import Note
from tensorflow.keras.models import load_model

from midilib.featuring.notebased import NoteBasedFeaturer
from midilib.featuring.util import normalize_song


class PitchDependentWaitModel:

    def __init__(self, notebased_featurer, pitch_model, wait_model):
        """

        Parameters
        ----------
        notebased_featurer : NoteBasedFeaturer
        pitch_model : keras.models.Model
        wait_model : keras.models.Model
        """
        self.notebased_featurer = notebased_featurer
        self.pitch_model = pitch_model
        self.wait_model = wait_model

    def dump(self, path):
        os.mkdir(path)
        self.notebased_featurer.to_path(
            os.path.join(path, 'pitch_dependent_wait_featurer'))
        self.pitch_model.save(
            os.path.join(path, 'pitch_model'))
        self.wait_model.save(
            os.path.join(path, 'pitch_dependent_wait_model'))

    @classmethod
    def load(cls, path):
        notebased_featurer = NoteBasedFeaturer.from_path(
            os.path.join(path, 'pitch_dependent_wait_featurer'))
        pitch_model = load_model(
            os.path.join(path, 'pitch_model'))
        pitch_dependent_wait_model = load_model(
            os.path.join(path, 'pitch_dependent_wait_model'))

        return cls(notebased_featurer, pitch_model, pitch_dependent_wait_model)

    def next_note(self, notes):
        """


        Parameters
        ----------
        notes

        Returns
        -------
        prettymidi.Note
        """
        notes = normalize_song(notes)
        fnotes = self.notebased_featurer.feature_notes(notes)
        for fnote in fnotes:
            fnote.calculate_features()
        sequence = self.notebased_featurer.extract_sequence_for_prediction(fnotes)
        pitch = np.argmax(self.pitch_model.predict(sequence)[0])
        wait = self.wait_model.predict([sequence, [pitch]])[0]

        # Convert back to note
        pitch = self.notebased_featurer.unmap_pitch(pitch)
        start = notes[-1].start + wait

        return Note(pitch=pitch, start=start, end=start + 0.5, velocity=128)

