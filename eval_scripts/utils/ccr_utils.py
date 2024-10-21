import torch
from transformers import BertTokenizer, BertForMaskedLM


####
# some utils for beam searching
sm = torch.nn.Softmax(dim=1) # In order to construct word probabilities, we will employ softmax.
torch.set_grad_enabled(False) # Since we are not training, we disable gradient calculation.

def join_tokens(tokens):
    """
    Joins a list of tokens into a string. Spaces are added between tokens unless a token starts with '##',
    in which case it is joined with the previous token without a space.
    
    :param tokens: List of tokens to be joined.
    :return: A single string with tokens joined according to the specified rules.
    """
    if not tokens:
        return ""
    
    # Initialize the result with the first token
    result = tokens[0]
    
    # Iterate over the tokens starting from the second one
    for token in tokens[1:]:
        # If the token starts with '##', join it with the previous token without a space
        if token.startswith("##"):
            result += token[2:]  # Remove '##' before joining
        else:
            # Otherwise, add a space before joining the token
            result += " " + token
    if result.startswith('##'):
       result = result[2:]
    return result

# Get top 5 suggestions for each masked position:
def argkmax(array, k, prefix='', tok=None, dim=0): # Return indices of the 1st through kth largest values of an array
  indices = []
  new_prefixes = []
  added = 0
  ind = 1
  while added < k:
    if ind > len(array[0]):
      break
    val = torch.kthvalue(-array, ind, dim=dim).indices.numpy().tolist()
    if prefix != '':
      cur_tok = tok.convert_ids_to_tokens(val[0]).replace('##', '')
      trunc_prefix = prefix[:min(len(prefix), len(cur_tok))]
      if not cur_tok.startswith(trunc_prefix):
        ind += 1
        continue
    else:
      cur_tok = ''
    indices.append(val)
    if len(cur_tok) >= len(prefix):
      new_prefixes.append('')
    else:
      new_prefixes.append(prefix[len(cur_tok):])
    ind += 1
    added += 1
  return torch.tensor(indices), new_prefixes

# gets n predictions / probabilities for a single masked token , by default, the first masked token
def get_n_preds(token_ids, n, prefix, masked_ind, fill_inds, cur_prob=1, model=None, tok=None):
  mask_positions = (token_ids.squeeze() == tok.mask_token_id).nonzero().flatten().tolist()
  for i in range(len(fill_inds)):
    token_ids.squeeze()[mask_positions[i]] = fill_inds[i]

  greekbert = model
  device = model.device
  logits = greekbert(token_ids.to(device)).logits.squeeze(0).cpu()
  mask_logits = logits[[[masked_ind]]]
  probabilities = sm(mask_logits)
  arg1, prefixes = argkmax(probabilities, n, prefix, tok, dim=1)
  suggestion_ids = arg1.squeeze().tolist()
  n_probs = probabilities.squeeze()[suggestion_ids]
  n_probs = torch.mul(n_probs, cur_prob).tolist()
  new_fill_inds = [fill_inds + [i] for i in suggestion_ids]
  return tuple(zip(new_fill_inds, n_probs, prefixes))

def beam_search(token_ids, beam_size, prefix='', breadth=100, model=None, tok=None):
  mask_positions = (token_ids.detach().clone().squeeze() == tok.mask_token_id).nonzero().flatten().tolist()

  num_masked = len(mask_positions)
  cur_preds = get_n_preds(token_ids.detach().clone(), beam_size, prefix, mask_positions[0], [], model=model, tok=tok)

  for i in range(num_masked - 1):
    candidates = []
    for j in range(len(cur_preds)):
      candidates += get_n_preds(token_ids.detach().clone(), breadth, cur_preds[j][2], mask_positions[i + 1], cur_preds[j][0], cur_preds[j][1], model=model, tok=tok)
    candidates.sort(key=lambda k:k[1],reverse=True)
    if i != num_masked - 2:
      cur_preds = candidates[:beam_size]
    else:
      cur_preds = candidates[:breadth]

  return cur_preds

def no_punc(token):
    valid_letters =  'ςερτυθιοπλκξηγφδσαζχψωβνμ#'
    for letter in token:
        if letter not in valid_letters:
            return False
    return True

def is_lev1(word1, word2):
    if word1 == word2:
       return False
    len1, len2 = len(word1), len(word2)
    if abs(len1 - len2) > 1:  # Quick check on length difference
        return False

    # Identify the shorter and longer string
    if len1 > len2:
        longer, shorter = word1, word2
    else:
        longer, shorter = word2, word1

    found_difference = False
    i, j = 0, 0
    while i < len(longer) and j < len(shorter):
        if longer[i] != shorter[j]:
            if found_difference:  # More than one mismatch
                return False
            found_difference = True
            # If lengths are different, move pointer of longer string
            if len1 != len2:
                i += 1
                continue
        i += 1
        j += 1

    return True

def replace_and_duplicate(lst, target, n):
    return [item for x in lst for item in ([x] if x != target else [target] * n)]

def get_lev1_suggestions(masked_text, gt_word, model, tok):
  if gt_word.startswith('##'):
     gt_word = gt_word[2:]
  overall_sugs = []
  for num_masks in range(1, min(len(gt_word), 3) + 1):
    text = replace_and_duplicate(masked_text, tok.mask_token_id, num_masks)
    tokens = torch.tensor([text])
    # print(text)
    sugs = beam_search(tokens, 10, '', model=model, tok=tok)
    for s in sugs:
      # if not first_tok.startswith('συν'):
        # continue
      converted = tok.convert_ids_to_tokens(s[0])
      word = join_tokens(converted)
      word_prob = s[1]
      if abs(len(word) - len(gt_word)) <= 1.1: # lol im scared of abs edge cases
         overall_sugs.append((word, word_prob))

  sorted_list = sorted(overall_sugs, key=lambda x: x[1])
  for pair in sorted_list:
     if is_lev1(gt_word, pair[0]):
        return pair[0], pair[1]
  return None, None
######


######
# code for calculating ccr
valid_letters =  '#ςερτυθιοπλκξηγφδσαζχψωβνμ'

# Softmax for probability calculation
sm = torch.nn.Softmax(dim=-1)
torch.set_grad_enabled(False)

def calculate_chance_confidence_ratio(sentence, position, model, tokenizer):
    char_token_ids = tokenizer(sentence, max_length=512, truncation=True)['input_ids']
    original_token_id = char_token_ids[position]  # Save the ground truth token id
    print(f"original: {tokenizer.convert_ids_to_tokens([original_token_id])}")

    # Mask the current token
    char_token_ids[position] = tokenizer.encode('[MASK]')[1]
    input_ids = torch.tensor(char_token_ids, dtype=torch.long).unsqueeze(0).to(model.device)
    # Get model output
    outputs = model(input_ids)
    logits = outputs.logits

    # Calculate probabilities
    probabilities = sm(logits[0, position, :])

    top_beamed_tok, top_beamed_prob = get_lev1_suggestions(char_token_ids, tokenizer.convert_ids_to_tokens([original_token_id])[0], model, tokenizer)
    print(f"sug: {top_beamed_tok}")
    if not top_beamed_tok:
        top_beamed_prob = 0
        char_token_ids[position] = original_token_id

    # Get the probability of the ground truth token
    ground_truth_prob = probabilities[original_token_id].item()

    # Calculate the chance-confidence ratio
    ratio = top_beamed_prob / ground_truth_prob if ground_truth_prob != 0 else float('inf')

    # Restore the original token id for the next iteration
    char_token_ids[position] = original_token_id

    return ratio

def get_indices(index, string_ls, tokenizer):
    input_ids = tokenizer(string_ls)['input_ids']
    sum = 0
    for i in range(index):
        sum += (len(input_ids[i]) - 2)
    return list(range(sum + 1, sum + len(input_ids[index]) - 1)) ## plus one to avoid CLS

def get_ccr_for_word(sentence, word_index, model, tokenizer):
    inds = get_indices(word_index, sentence.split(), tokenizer)
    ccrs = [calculate_chance_confidence_ratio(sentence, ind, model, tokenizer) for ind in inds]
    return max(ccrs)
######

def main():
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  tok = BertTokenizer.from_pretrained('YOUR_PATH_HERE')
  model = BertForMaskedLM.from_pretrained('YOUR_PATH_HERE').to(device)
  model.eval()
  text = f"αλλ ' εστω μητε δανειον , μητε παρακατα{tok.mask_token} , μητ ' αλλο τι των επαναστρεφοντων ο παρ ' εμου ειληφας , αλλα φιλοτιμια τις και διδασκαλια . τι ουν εμοι εφ ' οις πεφιλοτιμησαι μη ευγνωμονης ; τι μη την γλωτταν εν καιρω διδως ; πασης επιθυμουμεν ελληνων φωνης . ουχ ορας τον αερα οτι , θερους πολλακις ατμιδας απο της γης ανενεγκων , χειμωνος μερος τι των ανενεχθεντων αντικατηνεγκεν , ου τοιουτον αποδιδους οιον ειληφει , αλλα πηξας και μεταβαλων , και υδωρ πεποιηκως την αναφοραν ; σε δε ουτε μεταβολην των λογων απαιτουμεν καλλιονα , ουτε τινα εργασιαν μετεωρον · αλλ ' αν αποδως οιον προειληφας , αυτο δη τουτο εχειν των νενομισμενων οιομεθα . αλλα μοι προς τους εμους λογους πεπονθατε , οιον τι προς τας επιστημονικας φωνας οι νεωτεροι · φριττουσι γαρ ατεχνως τα ξενα των ονοματων ακουοντες , τον » τομον « , τα » περισωτα « , επειδαν δ ' αυθις κατατολμησωσι των φωνων και θαμα τοις επιστημοσι προσεγγισωσι και φωνην εθισθωσι διδοναι τε και λαμβανειν , αντι του εκπληττεσθαι , και καταφρονουσιν · ωσπερ και αυτος πεπονθα , ωρων εστιν οτε ακουων ασυμμετρον μεγεθος , ειτα δη ευρηκως , επιθαυμαζειν εαυτου κατεγινωσκον . τοιουτον δη σοι και το εμον εστι ."
