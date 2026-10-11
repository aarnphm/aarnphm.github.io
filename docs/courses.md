# courses directory

`content/courses/` is file-based routing. The path of a file sets its URL, the component that renders it, and the identity of its study data. The design history is in `docs/plans/courses.md` and `docs/plans/course-resources.md`. This page tells you where to put a file and what it becomes.

## two kinds of files

A course directory holds vendored files and authored files.

- **Vendored** files come from MIT OpenCourseWare through `pnpm vendor:ocw`. Do not edit them by hand. The next run writes over them. Change `quartz/scripts/vendor-ocw-course.ts` (pages, resources, home frontmatter) or `quartz/scripts/vendor-ocw-latex.ts` (LaTeX to `notes/`) and run the script again.
- **Authored** files are yours. The vendor script never writes to `lectures/`, `psets/` or `statements/`, and never deletes from them.

## routes

`<dir>` is the course directory name, for example `18.905-fall-2016`. Do not rename it. Flashcard review rows in D1 use the path as their `deck_slug`, so a rename orphans every review.

| file                                                 | URL                                       | renders                                                               | source                                         |
| ---------------------------------------------------- | ----------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------- |
| `index.md` (`layout: courses`)                       | `/courses/`                               | `CoursesIndex`: one row per course                                    | authored                                       |
| `<dir>/index.md` (`layout: course`)                  | `/courses/<dir>/`                         | `CourseHome`: masthead, rail, unit table, problem sets                | vendored                                       |
| `<dir>/pages/<name>/index.md`                        | `/courses/<dir>/pages/<name>/`            | OCW page with the course page head                                    | vendored                                       |
| `<dir>/pages/calendar/index.md`                      | `/courses/<dir>/pages/calendar/`          | `CourseCalendar`: month view of units and problem sets                | vendored, or authored when OCW has no calendar |
| `<dir>/resources/<id>/index.md` (`layout: resource`) | `/courses/<dir>/resources/<id>/`          | `CourseResource`: PDF reader, video, transcript, collection           | vendored                                       |
| `<dir>/notes/<nn>.md`                                | `/courses/<dir>/notes/<nn>`               | lecture text from LaTeX, unit bar, workspace                          | vendored (LaTeX courses)                       |
| `<dir>/notes/pset-<nn>.md`                           | `/courses/<dir>/notes/pset-<nn>`          | problem set text, desk                                                | vendored (LaTeX courses)                       |
| `<dir>/lectures/index.md` (`layout: study-log`)      | `/courses/<dir>/lectures/`                | `CourseStudyLog`: read and problem dates                              | authored                                       |
| `<dir>/lectures/<nn>.md`                             | `/courses/<dir>/lectures/<nn>`            | your notes, unit bar, workspace                                       | authored                                       |
| `<dir>/lectures/<nn>.fc`                             | `/courses/<dir>/lectures/<nn>/flashcards` | the unit deck                                                         | authored                                       |
| `<dir>/psets/<nn>.md`                                | `/courses/<dir>/psets/<nn>`               | your solutions, desk states in `problems:`                            | authored                                       |
| `<dir>/statements/index.md` (`layout: statements`)   | `/courses/<dir>/statements/`              | `CourseStatements`: theorems, definitions and the cards that use them | authored                                       |

The slug patterns live in `quartz/util/course.ts` (`unitOf`, `coursePageOf`, `resourceOf`). `FolderContent` picks the folder views from `pageLayout`; the frontmatter transformer turns `layout:` into `pageLayout`. `Content` adds the workspace to `lectures/<nn>` and `notes/<nn>`.

## numbers

A unit is a two-digit stem: `07`, not `7` and not `07-homotopy`. Put the title in frontmatter. The number must match `n` in the `course.units` list of the course home, because that list joins the notes, decks, calendar and study log. Problem sets use the number from the OCW resource id (`pset3`, `problemset_3`).

Keep files flat. `lectures/07.md` and `lectures/07.fc` are a pair. `lectures/07/index.fc` would give the deck slug `…/index/flashcards`.

## the course home

`<dir>/index.md` carries the course data in frontmatter:

```yaml
layout: course
course:
  number: '18.905'
  unitKind: lecture # or week, session
  parts: [...] # optional groups
  trailing: [...] # rows after the last unit (exams)
  units:
    - { n: 1, title: '...', keyDates: [...], resources: [...], due: [<pset resource id>] }
```

The vendor script builds `units` from the OCW calendar table. Without a calendar it uses numbered lecture-note resources, then the session table of a readings page. A course without units falls back to the plain `A|L` layout and gets none of the study views. Problem sets named in `due` show on the home and the calendar at that unit. Other `pset` resources show on the home with no due date.

## a new course

1. `pnpm vendor:ocw <ocw course url>` writes `content/courses/<dir>/`. To run again from the vendored `data.json` files, give the path: `pnpm vendor:ocw content/courses/<dir>`.
2. Check that `<dir>/index.md` has `layout: course` and a `units` list. If not, fix the vendor script, not the file.
3. Add the authored pages every course has (copy them from `18.905-fall-2016`):
   - `lectures/index.md` with `layout: study-log` and `study: { read: {}, problems: {} }`
   - `statements/index.md` with `layout: statements`
   - `pages/calendar/index.md` when OCW ships no calendar (see `18-156-projection-theory-spring-2025`)
4. Run `python3 -I docs/verification/course-parity/audit.py content/courses` and compare the new course with the others.

## studying a unit

- **Notes.** On a unit with no `lectures/<nn>.md`, the PDF strip offers `start notes`. It builds the note and a `.fc` deck from your PDF marks. In dev it writes both files; on the built site it copies them. Or write `lectures/<nn>.md` by hand.
- **Workspace.** Lecture pages (`lectures/<nn>`, `notes/<nn>`) open the workspace: source, scratchpad, notes. Signed-in notes are private marks on the page slug, so notes on `notes/03` and on `lectures/03` stay apart (`docs/course-workspace.md`).
- **Cards.** Write `lectures/<nn>.fc`. Add `ID:` pins to keep a card's review history when you reword it. In cloze lines write `\lbrack \rbrack` and `\lvert \rvert`.
- **Study log.** The `read` and `problems` dates in `lectures/index.md` place units on the calendar and set the frontier. In dev the buttons post them to the study endpoint (`quartz/scripts/course-study-server.ts`).
- **Problem sets.** `psets/<nn>.md` holds your solutions and the desk states (`problems:`). The desk needs problem text: from `notes/pset-<nn>` in a LaTeX course, or from the problems in your `psets/<nn>.md`.
- **Statements.** The statements page lists callouts from `notes/<nn>` and bold leads from `lectures/<nn>`. It stays empty until a unit has one of them.

## checks

- Use the running `quartz/scripts/dev.ts` watcher. Wait for `build:ready`. If a new page is missing from `public/` after that, the watcher missed the add event; touch the file once.
- `docs/verification/pset-ref-resolution/check-course-links.py` checks statement and equation links in LaTeX notes.
- `pnpm test worker/pdf-marks.test.ts` covers the mark sources the workspace uses.
