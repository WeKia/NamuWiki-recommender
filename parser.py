import json
import pandas as pd
import traceback
import time
import multiprocessing as mp
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, help='Path for namu wiki dump json file')
    parser.add_argument('--output', type=str, default='json_parsed.csv',
                        help='Path for output file')
    parser.add_argument('--proc_num', type=int, help='Number of parsing processor default is -1 that is max core of cpu', default=-1)

    args = parser.parse_args()

    return args

class worker(mp.Process):
    def __init__(self, dq, id, wq, rq):
        super(worker, self).__init__(daemon=True)
        self.wq = wq
        self.id = id
        self.dq = dq
        self.rq = rq

    def run(self):
        while True:
            data = self.dq.get()

            # Remove special parenthesis that is special commands
            # ex. [include(~~)], [* ~~]
            copied_text = data['text'] + ""
            new_text = ""
            markup = True

            link = set()
            category = []

            # For debug
            inner_commands = []

            # Delete useless syntax (Content of them is useless)
            # remove some syntaxes
            # look https://namu.wiki/w/%EB%82%98%EB%AC%B4%EC%9C%84%ED%82%A4:%EB%AC%B8%EB%B2%95%20%EB%8F%84%EC%9B%80%EB%A7%90

            syntax = ['[youtube', '[kakaotv', '[nicovideo', '<-', '[include', '[age',
                      '<table', '<width', '<height', '<bgcolor', '<col', '<row', '<(>',
                      '<:>', '<)>', '<^|', '<|', '<v|', '[date]', '[datetime]', '[dday', '[clearfix]', '[br]', '[목차]',
                      '||', '\'\'\'',  '---------', '--------', '-------', '------', '-----', '----',
                      '======', '=====', '====', '===', '==', '======#', '=====#', '====#', '===#', '==#',
                      '#======', '#=====', '#====', '#===', '#==', '>', ' 1.', ' *', ' a.', ' A.', '</font>'
                      '__', '\'\'', '~~', '--', '^^', ',,', '<#', '[ 펼치기 · 접기 ]']

            partner = {
                '[youtube' : ']',
                '[kakaotv' : ')]',
                '[nicovideo' : ')]',
                '<-' : '>',
                '<#' : '>',
                '<table' : '>',
                '[include' : ')]',
                '[age' : ']',
                '<width' : '>',
                '<height' : '>',
                '<bgcolor' : '>',
                '<col' : '>',
                '<row' : '>',
                '<^|' : '>',
                '<|' : '>',
                '<v|' : '>',
                '[dday' : ']',
            }

            # Since {{{ is non-markup command but some command uses same end command }}}
            # So to seperate them use stack
            stack = []
            offset = 0
            content = ''
            islink = False

            try:
                while offset < len(copied_text):
                    match = False
                    # Check Each words are commands

                    if copied_text[offset : offset + 9] == '#redirect' and markup:
                        new_text  = copied_text

                        linkto = copied_text[offset+10:]

                        linkto = linkto.replace('../', '')

                        # To process #
                        # Look https://namu.wiki/w/%EB%82%98%EB%AC%B4%EC%9C%84%ED%82%A4:%EB%AC%B8%EB%B2%95%20%EB%8F%84%EC%9B%80%EB%A7%90/%EC%8B%AC%ED%99%94#%ED%95%98%EC%9D%B4%ED%8D%BC%EB%A7%81%ED%81%AC%20%EC%8B%9C%20%EC%A3%BC%EC%9D%98
                        # For more detail
                        right_most = linkto.rfind('#')

                        while right_most > 0 and (linkto[right_most - 1] == '\\'):
                            right_most = linkto.rfind('#', 0, right_most)

                        if right_most != -1:
                            linkto = linkto[:right_most]

                        linkto = linkto.replace('\\#', '#')
                        link.update([linkto.strip()])

                        break


                    if copied_text[offset : offset + 3] == '{{{' and markup:
                        inner_command = copied_text[offset + 3]
                        inner_commands.append(inner_command)

                        if inner_command in ['+', '-', '#']:
                            stack.append('{{{*')
                            if markup:
                                while copied_text[offset] != ' ' and copied_text[offset:offset+3]!='}}}':
                                    offset += 1

                                if copied_text[offset] == ' ':
                                    offset += 1
                                elif copied_text[offset:offset+3]=='}}}':
                                    offset += 3
                                else:
                                    raise

                                if copied_text[offset : offset +5] == 'style':
                                    count = 0
                                    while count < 2:
                                        if copied_text[offset] == '\"' or copied_text[offset] == '\n':
                                            count += 1
                                        offset += 1
                        else:
                            stack.append('{{{')
                            # Prevent offset from overflow length of copied_text
                            offset += 3

                            if offset >= len(copied_text):
                                break

                            markup = False
                        match = True

                    if markup and (copied_text[offset: offset + 2] == '[[') :

                        if copied_text[offset+2:offset+5] == '파일:':
                            while copied_text[offset:offset+2] != ']]' :
                                if copied_text[offset] == '\\':
                                    offset += 1
                                offset += 1

                            offset += 2

                            if offset >= len(copied_text):
                                break

                        elif copied_text[offset + 2:offset + 5] == '분류:':
                            offset += 5
                            cat = ''

                            while copied_text[offset:offset+2] != ']]':
                                if copied_text[offset] == '\\':
                                    offset += 1
                                cat += copied_text[offset]
                                offset += 1

                                if offset >= len(copied_text):
                                    break

                            category.append(cat.replace('#blur', ''))

                            offset += 2

                            if offset >= len(copied_text):
                                break


                        else:
                            content = ''
                            offset += 2

                            if offset >= len(copied_text):
                                break

                            islink = True


                        match = True

                    if copied_text[offset : offset + 3] == '}}}' :
                        # Some }}} not match properly, just ignore them.
                        if len(stack) <= 0:
                            last_command = None
                        else:
                            last_command = stack.pop(-1)

                        if last_command == '{{{':
                            markup = True

                        if markup:
                            offset += 3

                            if offset >= len(copied_text):
                                break

                            match = True

                    if copied_text[offset : offset + 2] == ']]':
                        islink = False

                        # Offset for content checker
                        offset2 = 0
                        display = ''
                        linkto = ''

                        # To seperate display and link
                        do_display = False

                        cont_len = len(content)
                        while offset2 < cont_len:
                            if content[offset2] == '|' and not do_display:
                                display = ''
                                do_display = True
                                offset2 += 1

                            if offset2 >= cont_len:
                                break

                            display += content[offset2]
                            linkto += content[offset2] if not do_display else ''

                            offset2 += 1

                        linkto = linkto.replace('../', '')

                        # To process #
                        # Look https://namu.wiki/w/%EB%82%98%EB%AC%B4%EC%9C%84%ED%82%A4:%EB%AC%B8%EB%B2%95%20%EB%8F%84%EC%9B%80%EB%A7%90/%EC%8B%AC%ED%99%94#%ED%95%98%EC%9D%B4%ED%8D%BC%EB%A7%81%ED%81%AC%20%EC%8B%9C%20%EC%A3%BC%EC%9D%98
                        # For more detail
                        right_most = linkto.rfind('#')

                        while right_most > 0 and linkto[right_most - 1] == '\\':
                            right_most = linkto.rfind('#', 0, right_most)

                        if right_most != -1:
                            linkto = linkto[:right_most]

                        linkto = linkto.replace('\\#', '#')

                        if not (linkto[:8] == "https://" or linkto[:7] == "http://"):
                            link.update([linkto])

                        new_text += display if len(display) > 0 else linkto

                        offset += 2
                        match = True

                        if offset >= len(copied_text):
                            break

                    if markup:
                        if copied_text[offset] == '\\':
                            offset += 1

                            if islink and offset < len(copied_text) and copied_text[offset] == "#":
                                content += '\\'

                        # compare every commands
                        for s in syntax:
                            if copied_text[offset : offset + len(s)] == s:
                                offset += len(s)

                                if partner.get(s) is not None:
                                    p = partner[s]
                                    try :
                                        prev_offset = offset

                                        while copied_text[offset : offset + len(p)] != p:
                                            if copied_text[offset] == '\\':
                                                offset += 1

                                            offset += 1

                                            if offset >= len(copied_text):
                                                break

                                        offset += len(p)

                                        if offset >= len(copied_text):
                                            offset = prev_offset
                                            break

                                    except:
                                        print(s)

                                        raise

                                if s == '[br]':
                                    if islink:
                                        content += ' '
                                    else:
                                        new_text += ' '

                                match = True
                                break

                    if offset >= len(copied_text):
                        break

                    if not match:
                        if islink:
                            content += copied_text[offset]
                        else:
                            new_text += copied_text[offset]

                        offset += 1

                link = list(link)

                new_text = new_text.replace('\n', ' ')
                new_text = new_text.replace('\r', ' ')

                while new_text.find("  ") >= 0:
                    new_text = new_text.replace('  ', ' ')

                diction = {'title': data['title'],
                           'text': new_text[:512],
                           'category': [category],
                           'links': [link],
                           'contributors': [data['contributors']]}

                self.rq.put(diction)
            except:
                print(new_text.replace('\n', ' '))
                print('---------------------------------------------')
                print(copied_text.replace('\n', ' '))
                traceback.print_exc()

            self.wq.put(self.id)





def main(args):
    f = open(args.json_path, 'r', encoding='utf8')

    if args.proc_num == -1:
        process_num = mp.cpu_count() - 1
    else:
        process_num = args.proc_num

    frame = None
    doc = ""
    block_size = 4096

    # queue for non running process
    work_queue = mp.Queue(maxsize=process_num)
    # queue for processed data
    result_queue = mp.Queue()
    pool = []
    for i in range(process_num):
        data_queue = mp.Queue()
        w = worker(data_queue, i, work_queue, result_queue)
        w.start()
        pool.append(data_queue)
        work_queue.put(i)

    start_time = time.time()

    processed_docs = 0

    f.seek(0, 2)
    eof = f.tell()
    f.seek(0)

    read = 0

    while read < eof:
        # Since json file is too large, read it successively
        offset = 0
        doc_start = 0
        block = f.read(block_size)
        read += block_size

        while True:
            # Each json element is document, Object is parsing elements
            # if read nothing at previous block, find start of document element
            if doc == "":
                offset = doc_start = block.find("{", offset)

                if doc_start == -1:
                    break

            # Find end of json element
            # Assume that element is end with ]},
            doc_end = block.find("]},", offset)

            # If end of element is in block
            # Parse json to dict and init doc
            if doc_end != -1:
                offset = doc_end + 2

                doc += block[doc_start:doc_end + 2]

                if doc.find('\"contributors\"') == -1:
                    # every json element ends after contributors So check whether contributors appear
                    offset += 1
                    doc += ','
                    continue

                #print(doc)

                try:
                    data = json.loads(doc)

                    if True:
                        i = work_queue.get()

                        dq = pool[i]

                        dq.put(data)

                        processed_docs += 1

                        if processed_docs % 100 == 0:
                            print(f'docs proccessed {processed_docs} times!')
                            print(f'time elapsed {time.time() - start_time}')

                except:
                    print(doc)
                    traceback.print_exc()
                    
                    raise


                doc = ""
            else:
                # If end of element is not in this block
                # If the case end of element }], is divided into two blocks
                # Fetch next block
                if block[-1] == '}' or block[-1] == ']':
                    block += f.read(block_size)
                    read += block_size

                    if read >= eof:
                        break

                    continue

                doc += block[doc_start:]
                break

    print('process done')

    while not work_queue.full():
        continue

    print('do concatenate')

    while not result_queue.empty():
        diction = result_queue.get()

        data = pd.DataFrame(data=diction)

        if frame is None:
            frame = data
        else:
            frame = pd.concat([frame, data])

    print('Write csv')

    frame.to_csv(args.output, index=False,encoding='utf-8-sig')


if __name__ == '__main__':
    args = get_parser()
    main(args)