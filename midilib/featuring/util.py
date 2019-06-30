def normalize_song(song, chord_time=0.05):
    song.sort(key=lambda note: note.start)

    # Songs all start at t=0
    first_note_start = min(note.start for note in song)
    last_note_time = -1  # every song begins after 1 second delay on a out-of-range note
    for note in song:
        note.start -= first_note_start
        note.end -= first_note_start
        assert note.start >= last_note_time  # check that notes are sorted

        if note.start <= last_note_time + chord_time:  # simple algorithm to detect chords
            note.start = last_note_time
        else:
            last_note_time = note.start

    song.sort(key=lambda note: (note.start, note.pitch))

    return song


def constrain_pitch(pitch, min_pitch, max_pitch):
    max_pitch -= 1  # exclusive
    if pitch < min_pitch:
        pitch = pitch % 12 + 12 * (min_pitch // 12) + 12
        if pitch >= min_pitch + 12:
            pitch -= 12
        assert min_pitch <= pitch < min_pitch + 12
        return pitch
    if pitch > max_pitch:
        pitch = pitch % 12 + 12 * (max_pitch // 12) - 12
        if pitch <= max_pitch - 12:
            pitch += 12
        assert max_pitch - 12 < pitch <= max_pitch
        return pitch
    return pitch
