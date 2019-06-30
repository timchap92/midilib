

def dump_to_gs(fsongs, name, version):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('verbatim')
    timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    blob = bucket.blob(
        'midi/data/featured_songs/{n}_v{v}_{ts}.txt'.format(n=name, v=version, ts=timestamp))

    content = json.dumps(
        {'version': version, 'min_pitch': min_pitch, 'max_pitch': max_pitch}) + '\n'
    for fsong in registeredregistered_tqdm(fsongs):
        content += json.dumps([fnote.to_tuple() for fnote in fsong]) + '\n'

    blob.upload_from_string(content)


def load_featured_songs_from_gs(filename):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('verbatim')
    blob = bucket.blob('midi/data/featured_songs/{}'.format(filename))
    s = blob.download_as_string().decode('utf-8')
    lines = s.split('\n')
    print('Downloaded {lines} lines with metadata: {md}'.format(lines=len(lines) - 2, md=lines[0]))
    fsongs = []
    for line in registered_tqdm(lines[1:-1]):
        tpls = json.loads(line)
        fsongs.append([FeaturedNote.from_tuple(tpl) for tpl in tpls])

    return fsongs
