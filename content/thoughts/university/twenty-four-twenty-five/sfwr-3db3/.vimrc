filetype plugin indent on

syntax enable

" use vim-plug because of its minimalistic
"¯\_(ツ)_/¯

let data_dir = expand('~/.vim')
if empty(glob(data_dir . '/autoload/plug.vim'))
  silent execute '!curl -fLo '.data_dir.'/autoload/plug.vim --create-dirs  https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'
endif

" auto source vimrc
augroup vimrc
  autocmd!
  autocmd! BufWritePost $MYVIMRC source $MYVIMRC | echom "Reloaded $MYVIMRC"
augroup END

call plug#begin()
" tpope plugins =)
Plug 'tpope/vim-surround'
Plug 'tpope/vim-commentary'
Plug 'tpope/vim-vinegar'
Plug 'tpope/vim-sleuth'
Plug 'tpope/vim-apathy'

Plug 'nathanaelkane/vim-indent-guides'

" fzf with rg
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'

Plug 'rhysd/clever-f.vim'

Plug 'lambdalisue/suda.vim'
Plug 'airblade/vim-gitgutter'

" language packs
Plug 'sheerun/vim-polyglot'

" UI related
Plug 'godlygeek/tabular'
Plug 'rose-pine/vim'
call plug#end()

let mapleader=' '
let maplocalleader=','
let g:is_gui=has('gui_running')
let g:is_termguicolors = has('termguicolors') && !g:is_gui && $COLORTERM isnot# 'xterm-256color'
" tree style file explorer
let g:netrw_liststyle=3

" Minimal
set background=dark
colorscheme rosepine  " rosepine_dawn rosepine_moon

set number relativenumber

set backspace=indent,eol,start
set complete-=i

set nrformats-=octal

set laststatus=0
set ruler rulerformat=%70(%=%{GitStatus()}\ ›\ %y\ ›\ %{getfsize(@%)}B\ ›\ %P\ ›\ %l:%L%)
set wildmenu wildignore+=*.pyc,*.o,*.obj,*.swp,*.class,*.DS_Store,*.min.* wildmode=longest:full,full wildchar=<Tab>

set display+=lastline

set splitright splitbelow

set incsearch hlsearch
set noshowcmd showmode nojoinspaces
set notitle nowrap hidden
set clipboard=unnamed,unnamedplus
set mouse=a

set conceallevel=0
set foldmethod=manual "indent-syntax-manual
set shortmess+=Ic
set pastetoggle=<F2>

let WIDTH=2
set expandtab smarttab autoindent
let &shiftwidth=WIDTH
let &softtabstop=WIDTH
let &tabstop=WIDTH

" Performance tuning
set lazyredraw
set ignorecase smartcase

" Misc
set nobackup noswapfile nowritebackup
set undofile undolevels=9999
let &undodir=data_dir . '/undo'

set completeopt=menu,popup completepopup=highlight:Pmenu
set list listchars=tab:›\ ,nbsp:․,trail:·,extends:…,precedes:…
set autoread
set grepprg=rg\ -H\ --no-heading\ --vimgrep
set grepformat=%f:%l:%c:%m

set viminfo='200,<500,s32
set history=1000
set sessionoptions-=options viewoptions-=options

" Allow color schemes to do bright colors without forcing bold.
if &t_Co == 8 && $TERM !~# '^Eterm'
  set t_Co=16
endif

" Display an error message.
function! s:Warn(msg)
  echohl ErrorMsg
  echomsg a:msg
  echohl NONE
endfunction

" Call GitGutterGetHunkSummary to show list of git
function! GitStatus()
  let [a,m,r] = GitGutterGetHunkSummary()
  return printf('+%d ~%d -%d', a, m, r)
endfunction

" Command ':Bclose' executes ':bd' to delete buffer in current window.
" The window will show the alternate buffer (Ctrl-^) if it exists,
" or the previous buffer (:bp), or a blank buffer if no previous.
" Command ':Bclose!' is the same, but executes ':bd!' (discard changes).
" An optional argument can specify which buffer to close (name or number).
function! s:Bclose(bang, buffer)
  if empty(a:buffer)
    let btarget = bufnr('%')
  elseif a:buffer =~? '^\d\+$'
    let btarget = bufnr(str2nr(a:buffer))
  else
    let btarget = bufnr(a:buffer)
  endif
  if btarget < 0
    call s:Warn('No matching buffer for '.a:buffer)
    return
  endif
  if empty(a:bang) && getbufvar(btarget, '&modified')
    call s:Warn('No write since last change for buffer '.btarget.' (use :Bclose!)')
    return
  endif
  " Numbers of windows that view target buffer which we will delete.
  let wnums = filter(range(1, winnr('$')), 'winbufnr(v:val) == btarget')
  if !g:bclose_multiple && len(wnums) > 1
    call s:Warn('Buffer is in multiple windows (use ":let bclose_multiple=1")')
    return
  endif
  let wcurrent = winnr()
  for w in wnums
    execute w.'wincmd w'
    let prevbuf = bufnr('#')
    if prevbuf > 0 && buflisted(prevbuf) && prevbuf != btarget
      buffer #
    else
      bprevious
    endif
    if btarget == bufnr('%')
      " Numbers of listed buffers which are not the target to be deleted.
      let blisted = filter(range(1, bufnr('$')), 'buflisted(v:val) && v:val != btarget')
      " Listed, not target, and not displayed.
      let bhidden = filter(copy(blisted), 'bufwinnr(v:val) < 0')
      " Take the first buffer, if any (could be more intelligent).
      let bjump = (bhidden + blisted + [-1])[0]
      if bjump > 0
        execute 'buffer '.bjump
      else
        execute 'enew'.a:bang
      endif
    endif
  endfor
  execute 'bdelete'.a:bang.' '.btarget
  execute wcurrent.'wincmd w'
endfunction
command! -bang -complete=buffer -nargs=? Bclose call <SID>Bclose(<q-bang>, <q-args>)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" => Mapping
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Mapping
nnoremap ; :
imap jj <Esc>

" Thanks to Steve Losh for this liberating tip
" See http://stevelosh.com/blog/2010/09/coming-home-to-vim
if exists('plug')
  nnoremap / /\v
  vnoremap / /\v
endif

" We don't use arrow key in vim
noremap <Up> <Nop>
noremap <Down> <Nop>
noremap <Left> <Nop>
noremap <Right> <Nop>
inoremap <Up> <nop>
inoremap <Down> <nop>
inoremap <Left> <nop>
inoremap <Right> <nop>

" manual folding
inoremap <F5> <C-O>za
nnoremap <F5> za
onoremap <F5> <C-C>za
vnoremap <F5> zf
" Toggle show/hide invisible chars
nnoremap <leader>I :set list!<cr>
nnoremap \\ :let @/=''<CR>:noh<CR>
nnoremap <silent> <leader>p :%s///g<CR>
nnoremap <silent> <leader>i gg=G<CR>
nnoremap <leader># :g/\v^(#\|$)/d_<CR>
nnoremap <leader>b :ls<CR>:buffer<space>
nnoremap <leader>d :w !diff % -<CR>
nnoremap <leader>S :so $MYVIMRC<CR>
nnoremap <leader>l :set list! list?<CR>
nnoremap <leader>t :%s/\s\+$//e<CR>
" Remove the Windows ^M - when the encodings gets messed up
" for somereason bufread doesn't catch  it first
nnoremap <leader>m mmHmt:%s/<C-V><cr>//ge<cr>'tzt'm
nnoremap <leader>w :set wrap! wrap?<CR>
" When you press <leader>r you can search and replace the selected text
nnoremap <leader>ml :call AppendModeLine()<CR>
" Visual mode pressing * or # searches for the current selection
" Super useful! From an idea by Michael Naumann
vnoremap <silent> * :<C-u>call VisualSelection('', '')<CR>/<C-R>=@/<CR><CR>
vnoremap <silent> # :<C-u>call VisualSelection('', '')<CR>?<C-R>=@/<CR><CR>
" this is for transient
" Smart way to move between windows
map <C-j> <C-W>j
map <C-k> <C-W>k
map <C-h> <C-W>h
map <C-l> <C-W>l
" Opens a new tab with the current buffer's path
" Super useful when editing files in the same directory
nnoremap <LocalLeader>vs :vsplit<CR>
nnoremap <LocalLeader>hs :split<CR>
nnoremap <silent><nowait> <space>l :bNext<cr>
nnoremap <silent><nowait> <space>h :bprevious<cr>
nnoremap <silent><nowait> <C-x> :Bclose<CR>
vmap J :m '>+1<cr>gv=gv<CR>
vmap K :m '<-2<cr>gv=gv<CR>

" remove Ex mode
map Q <Nop>
" added yank to clipboard shortcut
noremap <M-Y> "*y
noremap <M-P> "*p
noremap <M-y> "+y
noremap <M-p> "+p
" use this when lightline is not in use for minimal
nnoremap <F2> :set invpaste paste?<CR>
imap <F2><C-O>:set invpaste paste?<CR>
" quick resize for split
nnoremap <silent> <leader>+ :exe "resize " . (winheight(0) * 3/2)<CR>
nnoremap <silent> <leader>- :exe "resize " . (winheight(0) * 2/3)<CR>
nnoremap <LocalLeader>p :PlugUpdate<cr>
if exists(':Tabularize')
    nmap <leader>a= :Tabularize /=<CR>
    vmap <leader>a= :Tabularize /=<CR>
    nmap <Leader>a: :Tabularize /:<CR>
    vmap <Leader>a: :Tabularize /:<CR>
    nmap <leader>a: :Tabularize /:\zs<CR>
    vmap <leader>a: :Tabularize /:\zs<CR>
endif

" insert mode mapping to trigger the :Tabular command when you type the character that you want to align.
inoremap <silent> <Bar>   <Bar><Esc>:call <SID>align()<CR>a

function! s:align()
    let p = '^\s*|\s.*\s|\s*$'
    if exists(':Tabularize') && getline('.') =~# '^\s*|' && (getline(line('.')-1) =~# p || getline(line('.')+1) =~# p)
        let column = strlen(substitute(getline('.')[0:col('.')],'[^|]','','g'))
        let position = strlen(matchstr(getline('.')[0:col('.')],'.*|\s*\zs.*'))
        Tabularize/|/l1
        normal! 0
        call search(repeat('[^|]*|',column).'\s\{-\}'.repeat('.',position),'ce',line('.'))
    endif
endfunction


" Append modeline after last line in buffer.
" Use substitute() instead of printf() to handle '%%s' modeline in LaTeX
" files.
function! AppendModeLine()
    let l:modeline = printf('vim: set ft=%s ts=%d sw=%d tw=%d %set :',
                \ &filetype, &tabstop, &shiftwidth, &textwidth, &expandtab ? '' : 'no')
    let l:modeline = substitute(&commentstring, '%s', l:modeline, '')
    call append(line('$'), l:modeline)
endfunction

function! CmdLine(str)
    call feedkeys(':' . a:str)
endfunction

function! VisualSelection(direction, extra_filter) range
    let l:saved_reg = @"
    execute 'normal! vgvy'

    let l:pattern = escape(@", "\\/.*'$^~[]")
    let l:pattern = substitute(l:pattern, '\n$', '', '')

    if a:direction ==? 'gv'
        call CmdLine("Ack '" . l:pattern . "' " )
    elseif a:direction ==? 'replace'
        call CmdLine('%s' . '/'. l:pattern . '/')
    endif

    let @/ = l:pattern
    let @" = l:saved_reg
endfunction

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" => Plugins settings
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

let g:suda_smart_edit = 1
let g:suda#prompt = 'Mot de passe: '

let g:vim_markdown_folding_disabled = 1
let g:vim_markdown_conceal = 0
let g:vim_markdown_conceal_code_blocks = 0

let g:indent_guides_default_mapping = 0
let g:indent_guides_enable_on_vim_startup = 1
let g:indent_guides_start_level = 2
let g:indent_guides_exclude_filetypes = ['help', 'startify', 'man', 'rogue']

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" fzf
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Path completion with custom source command
inoremap <expr> <c-x><c-f> fzf#vim#complete#path('fd')
inoremap <expr> <c-x><c-f> fzf#vim#complete#path('rg --files')

" Word completion with custom spec with popup layout option
inoremap <expr> <c-x><c-k> fzf#vim#complete#word({'window': { 'width': 0.2, 'height': 0.9, 'xoffset': 1 }})

" Default fzf layout
" - down / up / left / right
let g:fzf_layout = { 'window': { 'width': 0.9, 'height': 0.6 } }
let g:fzf_action = {
      \ 'ctrl-s': 'split',
      \ 'ctrl-v': 'vsplit'
      \ }

" Enable per-command history
" History files will be stored in the specified directory
" When set, CTRL-N and CTRL-P will be bound to 'next-history' and
" 'previous-history' instead of 'down' and 'up'.
nnoremap <silent><nowait> <Leader>f :Files <CR>
nnoremap <silent><nowait> <C-g> :Lines <CR>
nnoremap <silent><nowait> <Leader>b :Buffers <CR>
nnoremap <F7> :Snippets <CR>
nnoremap <F8> :Colors <CR>
nnoremap <silent><nowait> <C-p> :Maps <CR>

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" gitgutter
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

function! GitGutterNextHunkCycle()
  let line = line('.')
  silent! GitGutterNextHunk
  if line('.') == line
    1
    GitGutterNextHunk
  endif
endfunction

function! GitGutterPrevHunkCycle()
  let line = line('.')
  silent! GitGutterPrevHunk
  if line('.') == line
    normal! G
    GitGutterPrevHunk
  endif
endfunction

let g:gitgutter_enabled=1
let g:gitgutter_diff_args = '-w'
let g:gitgutter_grep = 'rg'
nmap ]h :call GitGutterNextHunkCycle()<CR>
nmap [h :call GitGutterPrevHunkCycle()<CR>

"vim: set ft=vim ts=2 sw=2 tw=78 et :
