import requests
import os
import pygame

SOUNDS_DIR = 'sounds'
if not os.path.exists(SOUNDS_DIR): os.mkdir(SOUNDS_DIR)

API_KEY = 'ENTER_KEY_HERE'
SEARCH_URL_TEMPLATE = "https://freesound.org/apiv2/search/text/?query=%s&token=" + API_KEY
SOUND_INSTANCE_URL_TEMPLATE = "https://freesound.org/apiv2/sounds/%s/?token=" + API_KEY

pygame.mixer.init()

def get_all_for_term(searchterm, category):
	basedir = os.path.join(SOUNDS_DIR, category.replace(' ', '_'))
	if not os.path.exists(basedir): os.mkdir(basedir)

	ignore_ids = set()
	ignore_file_dir = os.path.join(basedir, 'ignore.txt')
	keep_file_dir = os.path.join(basedir, 'keep.txt')

	if os.path.exists(ignore_file_dir):
		with open(ignore_file_dir) as f:
			for line in f:
				if not line.strip(): continue
				ignore_ids.add(int(line))

	if os.path.exists(keep_file_dir): #also ignore ones we've already accepted
		with open(keep_file_dir) as f:
			for line in f:
				if not line.strip(): continue
				ignore_ids.add(int(line))

	search_url = SEARCH_URL_TEMPLATE % searchterm.replace(' ', '%20')

	while 1:
		resp = requests.get(search_url)
		jsonresp = resp.json()

		for res in jsonresp['results']:
			if int(res['id']) in ignore_ids:
				print('skipping %s' % str(res['id']))
				continue

			sires = requests.get(SOUND_INSTANCE_URL_TEMPLATE % res['id']).json()
			sound_uri = sires['previews']['preview-lq-mp3']

			dest_uri = os.path.join(basedir, 'sound_' + str(res['id']) + '.mp3')
			if os.path.exists(dest_uri): continue
			os.system('curl -o %s %s' % (dest_uri, sound_uri))

			while 1:
				pygame.mixer.music.load(dest_uri)
				print('Playing "%s"...' % sires['name'])
				pygame.mixer.music.play()

				cmd = input('Keep? (y = yes, n = no, q = quit this sound, other = play again)')
				pygame.mixer.music.stop()
				if cmd == 'y':
					print('Keeping')
					os.remove(dest_uri)
					with open(keep_file_dir, 'a') as f:
						f.write(str(res['id']) + '\n')
					break
				elif cmd == 'n':
					print('Discarding')
					os.remove(dest_uri)
					with open(ignore_file_dir, 'a') as f:
						f.write(str(res['id']) + '\n')
					break
				elif cmd == 'q':
					print('Quitting')
					os.remove(dest_uri)
					return

		if 'next' in jsonresp and jsonresp['next']: 
			print('Proceeding to next page of results')
			search_url = jsonresp['next'] + '&token=' + API_KEY
		else: 
			print('No more results, done')
			break

while 1:
	term = input('Enter search term: ')
	category = input('Category: ')
	get_all_for_term(term, category)
