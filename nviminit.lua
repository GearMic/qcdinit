vim.keymap.set('n', '<F5>', ':w<CR>:!python src/main.py<CR>', {})
vim.keymap.set('n', '<F6>', ':w<CR>:!make<CR>', {})

-- vim.keymap.set('n', '<F5>', function()
--     vim.cmd.shell('python src/main.py')
-- end, {})

