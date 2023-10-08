" Yank to system clipboard
set clipboard=unnamed

" Go back and forward with Ctrl+O and Ctrl+I
" (make sure to remove default Obsidian shortcuts for these to work)
exmap back obcommand app:go-back
nmap <C-o> :back
exmap forward obcommand app:go-forward
nmap <C-i> :forwardv

" Mapping
nmap ; :
imap jj <Esc>

" Here's an example config that implements many of the features from vim-surround:
exmap surround_wiki surround [[ ]]
exmap surround_double_quotes surround " "
exmap surround_single_quotes surround ' '
exmap surround_brackets surround ( )
exmap surround_square_brackets surround [ ]
exmap surround_curly_brackets surround { }

" NOTE: must use 'map' and not 'nmap'
map [[ :surround_wiki
vunmap S
vmap S" :surround_double_quotes
vmap S' :surround_single_quotes
vmap Sb :surround_brackets
vmap S( :surround_brackets
vmap S) :surround_brackets
vmap S[ :surround_square_brackets
vmap S[ :surround_square_brackets
vmap S{ :surround_curly_brackets
vmap S} :surround_curly_brackets

" Split Window
exmap split_vertical obcommand workspace:split-vertical
nmap vs :split_vertical
exmap split_horizontal obcommand workspace:split-horizontal
nmap vh :split_horizontal

" Quit Buffer
exmap qq obcommand workspace:close

" Map to Open File
exmap open_file obcommand switcher:open 
nmap <Space>f :open_file

" Go Into Link
exmap goto_link obcommand editor:follow-link
nmap gd :goto_link

"""""""""""""""""""""""""""""""""""""""""""""""" "
"""""""""""" All Available Commands """""""""""" "
" :obcommand and <opt + cmd + i> to open console "
"""""""""""""""""""""""""""""""""""""""""""""""" "
" Available commands: editor:save-file
" editor:follow-link
" editor:open-link-in-new-leaf
" editor:open-link-in-new-split
" editor:open-link-in-new-window
" editor:focus-top
" editor:focus-bottom
" editor:focus-left
" editor:focus-right
" workspace:toggle-pin
" workspace:split-vertical
" workspace:split-horizontal
" workspace:toggle-stacked-tabs
" workspace:edit-file-title
" workspace:copy-path
" workspace:copy-url
" workspace:undo-close-pane
" workspace:export-pdf
" editor:rename-heading
" workspace:open-in-new-window
" workspace:move-to-new-window
" workspace:next-tab
" workspace:goto-tab-1
" workspace:goto-tab-2
" workspace:goto-tab-3
" workspace:goto-tab-4
" workspace:goto-tab-5
" workspace:goto-tab-6
" workspace:goto-tab-7
" workspace:goto-tab-8
" workspace:goto-last-tab
" workspace:previous-tab
" workspace:new-tab
" app:go-back
" app:go-forward
" app:open-vault
" theme:use-dark
" theme:use-light
" theme:switch
" app:open-settings
" app:show-release-notes
" markdown:toggle-preview
" workspace:close
" workspace:close-window
" workspace:close-others
" app:delete-file
" app:toggle-left-sidebar
" app:toggle-right-sidebar
" app:toggle-default-new-pane-mode
" app:open-help
" app:reload
" app:show-debug-info
" window:toggle-always-on-top
" window:zoom-in
" window:zoom-out
" window:reset-zoom
" file-explorer:new-file
" file-explorer:new-file-in-new-pane
" open-with-default-app:open
" file-explorer:move-file
" open-with-default-app:show
" editor:open-search
" editor:open-search-replace
" editor:focus
" editor:toggle-fold
" editor:fold-all
" editor:unfold-all
" editor:fold-less
" editor:fold-more
" editor:insert-wikilink
" editor:insert-embed
" editor:insert-link
" editor:insert-tag
" editor:set-heading
" editor:set-heading-0
" editor:set-heading-1
" editor:set-heading-2
" editor:set-heading-3
" editor:set-heading-4
" editor:set-heading-5
" editor:set-heading-6
" editor:toggle-bold
" editor:toggle-italics
" editor:toggle-strikethrough
" editor:toggle-highlight
" editor:toggle-code
" editor:toggle-blockquote
" editor:toggle-comments
" editor:toggle-bullet-list
" editor:toggle-numbered-list
" editor:toggle-checklist-status
" editor:cycle-list-checklist
" editor:insert-callout
" editor:swap-line-up
" editor:swap-line-down
" editor:attach-file
" editor:delete-paragraph
" editor:toggle-spellcheck
" editor:context-menu
" file-explorer:open
" file-explorer:reveal-active-file
" global-search:open
" switcher:open
" graph:open
" graph:open-local
" graph:animate
" backlink:open
" backlink:open-backlinks
" backlink:toggle-backlinks-in-document
" canvas:new-file
" canvas:export-as-image
" outgoing-links:open
" outgoing-links:open-for-current
" tag-pane:open
" daily-notes
" daily-notes:goto-prev
" daily-notes:goto-next
" insert-template
" insert-current-date
" insert-current-time
" note-composer:merge-file
" note-composer:split-file
" note-composer:extract-heading
" command-palette:open
" starred:open
" starred:star-current-file
" outline:open
" outline:open-for-current
" file-recovery:open
" editor:toggle-source
